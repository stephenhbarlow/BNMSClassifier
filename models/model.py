from transformers import AutoModel
from torch import nn
import numpy as np


class BertModel(nn.Module):

    def __init__(self, args):
        super(BertModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_ckpt, return_dict=False)
        self.drop = nn.Dropout(args.dropout_prob)
        self.out = nn.Linear(self.bert.config.hidden_size, args.n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        output = self.drop(pooled_output)

        return self.out(output)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
