import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RuARDDataset(Dataset):
    def __init__(self, text, target, config, is_test=False):
        self.text = text
        self.target = target
        self.is_test = is_test
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        #self.tokenizer.pad_token = 0
        self.max_len = config.max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        if self.is_test:
            return inputs
        else:
            return (inputs, torch.tensor(self.target[item], dtype=torch.float))
