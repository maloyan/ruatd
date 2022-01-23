import torch
import torch.nn as nn
from transformers import AutoModel


class BERTBaseUncased(nn.Module):
    def __init__(self, config):
        super(BERTBaseUncased, self).__init__()
        self.model = AutoModel.from_pretrained(config.model)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(1536 * 2, 1)

    def forward(self, **kwargs):
        # _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # bo = self.bert_drop(o2)
        # output = self.out(bo)
        # return output

        o1 = self.model(
            input_ids=kwargs["input_ids"].squeeze(1),
            attention_mask=kwargs["attention_mask"].squeeze(1),
            #token_type_ids=kwargs["token_type_ids"].squeeze(1),
        )["last_hidden_state"] #["pooler_output"] 
        mean_pooling = torch.mean(o1, 1)
        max_pooling = torch.max(o1, 1).values
        cat = torch.cat((mean_pooling, max_pooling), 1)
        bo = self.drop(cat)
        output = self.out(bo)
        return output
