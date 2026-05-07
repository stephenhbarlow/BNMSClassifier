import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, \
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def compute_scores(preds, label_set):
    scores = {'F1_MACRO': f1_score(label_set, preds, average="macro"),
              'F1_MICRO': f1_score(label_set, preds, average="micro"),
              'RECALL_MACRO': recall_score(label_set, preds, average="macro"),
              'RECALL_MICRO': recall_score(label_set, preds, average="micro"),
              'PRECISION_MACRO': precision_score(label_set, preds, average="macro"),
              'PRECISION_MICRO': precision_score(label_set, preds, average="micro")}

    return scores


def display_confusion_matrix(label_set, preds):
    cm = confusion_matrix(label_set, preds)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 22})
    disp = ConfusionMatrixDisplay(cm, display_labels=['M1 Negative','M1 Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('visualisations/confusion_matrix.pdf', bbox_inches='tight')
    plt.show()


def display_roc_curve(label_set, probs):
    
    fpr, tpr, _ = roc_curve(label_set, probs)
    roc_auc = auc(fpr, tpr)
    print(f"Val AUC: {roc_auc}")
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 22})
    plt.plot(fpr, tpr, marker='.', label='AUC = %0.2f' % roc_auc)
    plt.xticks(ticks=ticks)
    plt.yticks(ticks=ticks)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axis('square') 
    plt.legend(loc = 'lower right')
    plt.savefig('visualisations/auroc_curve.pdf', bbox_inches='tight')
    plt.show()
    