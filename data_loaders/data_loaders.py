from torch.utils.data import DataLoader
from data.datasets import ReportDataset


class BertDataLoader(DataLoader):

    def __init__(self, args, split, shuffle, drop_last=False):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = args.tokenizer
        self.max_len = args.max_len
        self.split = split

        if self.split == 'train':
            self.dataset = ReportDataset(self.args.train_data_dir, self.tokenizer, self.max_len)
        elif self.split == 'val':
            self.dataset = ReportDataset(self.args.val_data_dir, self.tokenizer, self.max_len)
        else:
            self.dataset = ReportDataset(self.args.test_data_dir, self.tokenizer, self.max_len)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'drop_last': drop_last
        }

        super().__init__(**self.init_kwargs)
