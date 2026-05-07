from transformers import AutoModel, AutoConfig, MegatronBertPreTrainedModel
from torch import nn
import numpy as np


class HFBertModel(MegatronBertPreTrainedModel):

    def __init__(self, args):
        super(HFBertModel, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.config)
        self.bert = AutoModel.from_pretrained(args.model_ckpt, return_dict=False)
        self.drop = nn.Dropout(args.dropout_prob)
        self.out = nn.Linear(self.bert.config.hidden_size, args.n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        output = self.drop(pooled_output)
        logits = self.out(output)

        return logits

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        # print("conservation: ", cam.sum())
        return cam