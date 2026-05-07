from tqdm import tqdm
import torch
import torch.nn.functional as nnf
import logging
import os
import numpy as np
from collections import defaultdict
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):

    def __init__(self, model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader):

        self.args = args
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = self.args.epochs
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.writer = SummaryWriter()

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self):

        progress = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader))
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        for i, d in progress:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)

            preds = torch.round(torch.sigmoid(outputs)).squeeze()
            loss = self.loss_fn(outputs, labels.unsqueeze(1))
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar('train_step_loss', loss, i+1)

        return correct_predictions.float() / len(self.train_dataloader.dataset), np.mean(losses)

    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        correct_predictions = 0
        y_test = []
        predictions = []
        probs = []

        with torch.no_grad():
            for i, d in enumerate(self.val_dataloader):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                preds = torch.round(torch.sigmoid(outputs)).squeeze()
                probability = torch.sigmoid(outputs)
                probs.append(probability)
                predictions.append(preds)
                y_test.append(labels)
                loss = self.loss_fn(outputs, labels.unsqueeze(1))
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

                self.writer.add_scalar('val_step_loss', loss, i+1)


        y_test = torch.cat(y_test).cpu().data.numpy()
        predictions = torch.cat(predictions).cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return correct_predictions.float() / len(self.val_dataloader.dataset), np.mean(losses), \
               y_test, predictions, probs

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_acc, train_loss = self.train_epoch()
            self.writer.add_scalar('train_epoch_loss', train_loss, epoch+1)
            self.writer.add_scalar('train_epoch_acc', train_acc, epoch+1)
            print(f'Train Loss: {train_loss} Accuracy: {train_acc}')
            val_acc, val_loss, y_test, predictions, probs = self.eval_model()
            self.writer.add_scalar('val_epoch_loss', train_loss, epoch+1)
            self.writer.add_scalar('val_epoch_acc', train_acc, epoch+1)
            print(f'Val   Loss: {val_loss} Accuracy: {val_acc}')

            # if val_acc > best_accuracy:
            torch.save(self.model.state_dict(),
                           f"{self.checkpoint_dir}/BERT-large"
                           f"{epoch + 1}epochs.bin")
                # best_accuracy = val_acc

        # print(compute_scores(predictions, y_test))
        print(classification_report(y_test, predictions))
        display_confusion_matrix(y_test, predictions)
        display_roc_curve(y_test, probs)
        self.writer.close()