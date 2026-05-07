import argparse
import numpy as np
import torch
import random

from evaluation.evaluate_model import EvaluateModel
from models.model import BertModel
from data_loaders.data_loaders import BertDataLoader


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

    # build model architecture

    model = BertModel(args)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # create data loader
    test_dataloader = BertDataLoader(args, split='test', shuffle=False)

    evaluator = EvaluateModel(model, args, test_dataloader)
    evaluator.eval()
    

if __name__ == '__main__':
    main()
