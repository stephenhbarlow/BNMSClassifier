import argparse
import numpy as np
import torch
import random

from evaluation.evaluate_model import EvaluateModel
from models.model import BertModel
from data_loaders.data_loaders import BertDataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, default='data/tnm_val2.csv',
                        help='the path to the directory containing the test data.')
    parser.add_argument('--model', type=str, default='results/gatortron_model_2epochs_batch4.bin')
    parser.add_argument('--tokenizer', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained tokenizer.')

    # Data loader settings
    parser.add_argument('--max_len', type=int, default=512, help='max length of sentence encoding')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=4, help='the number of samples for a batch')

    # Model settings (for Transformer)
    parser.add_argument('--model_ckpt', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained Transformer.')
    parser.add_argument('--n_classes', type=int, default=1, help='the number of output classes')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='the dropout rate of the output layer.')

    parser.add_argument('--seed', type=int, default=1234, help='.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create data loader
    test_dataloader = BertDataLoader(args, split='test', shuffle=False)

    model_list = ['results/BERT-base3epochs.bin', 'results/BioBERT-base5epochs.bin', 
                  'results/BioClinicalBERT4epochs.bin','results/gatortron_model_2epochs_batch4.bin']
    
    tokenizer_list = ['bert-base-cased', 'dmis-lab/biobert-base-cased-v1.2', 
                      'emilyalsentzer/Bio_ClinicalBERT', "UFNLP/gatortron-base"]
    
    probs_list = []
    label_list = []
    for i, _ in enumerate(model_list):
        args.tokenizer = tokenizer_list[i]
        args.model = model_list[i]
        args.model_ckpt = tokenizer_list[i]
        model = BertModel(args)
        model.load_state_dict(torch.load(args.model))
        model.eval()
        
        # create data loader
        test_dataloader = BertDataLoader(args, split='test', shuffle=False)

        evaluator = EvaluateModel(model, args, test_dataloader)
        _, _, y_test, _, probs = evaluator.eval_model()
        probs_list.append(probs)
        label_list.append(y_test)
    
    #set up plotting area
    plt.figure(0).clf()
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 16})

    bert_fpr, bert_tpr, _ = roc_curve(label_list[0], probs_list[0])
    plt.plot(bert_fpr, bert_tpr, marker='.', color="tab:gray", label="BERT")

    #fit gradient boosted model and plot ROC curve
    biobert_fpr, biobert_tpr, _ = roc_curve(label_list[1], probs_list[1])
    plt.plot(biobert_fpr, biobert_tpr, marker='.', color="tab:orange", label="BioBERT")

    #fit gradient boosted model and plot ROC curve
    bioclinbert_fpr, bioclinbert_tpr, _ = roc_curve(label_list[2], probs_list[2])
    plt.plot(bioclinbert_fpr, bioclinbert_tpr, marker='.', color="tab:green", label="BioClinicalBERT")

    #fit gradient boosted model and plot ROC curve
    gatortron_fpr, gatortron_tpr, _ = roc_curve(label_list[3], probs_list[3])
    plt.plot(gatortron_fpr, gatortron_tpr, marker='.', color="tab:blue", label="GatorTron")

    plt.xticks(ticks=ticks)
    plt.yticks(ticks=ticks)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axis('square')
    plt.legend(loc ='lower right', prop={'size': 18})
    plt.savefig('visualisations/multi__auroc_curve.pdf', bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    main()
