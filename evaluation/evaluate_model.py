import torch
import torch.nn.functional as nnf
import numpy as np
from models.metric import display_confusion_matrix, display_roc_curve
from sklearn.metrics import classification_report


class EvaluateModel(object):

    def __init__(self, model, args, test_dataloader):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = model.to(self.device)
        self.args = args
        self.test_dataloader = test_dataloader
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        correct_predictions = 0
        y_test = []
        predictions = []
        probs = []

        with torch.no_grad():

            for d in self.test_dataloader:
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

        y_test = torch.cat(y_test).cpu().data.numpy()
        predictions = torch.cat(predictions).cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return correct_predictions.float() / len(self.test_dataloader.dataset), np.mean(losses),  \
            y_test, predictions, probs

    def eval(self):

        test_acc, test_loss, y_test, predictions, probs = self.eval_model()
        print(f'Val   Loss: {test_loss} Accuracy: {test_acc}')
        print(classification_report(predictions, y_test))
        display_confusion_matrix(predictions, y_test)
        display_roc_curve(y_test, probs)
