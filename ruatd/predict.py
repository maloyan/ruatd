import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics, model_selection
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from ruatd.dataset import RuARDDataset
from ruatd.engine import eval_fn, train_fn
from ruatd.model import BERTBaseUncased


@hydra.main(config_path="config", config_name="bert")
def main(config: DictConfig):
    # df_valid = pd.read_csv(config.data.val)
    # df_valid.Class = df_valid.Class.apply(lambda x: 1 if x == "M" else 0)
    df_test = pd.read_csv(config.data.test)

    # valid_dataset = RuARDDataset(
    #     text=df_valid.Text.values, target=df_valid.Class.values, config=config
    # )

    # valid_data_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    #     pin_memory=config.pin_memory,
    # )

    test_dataset = RuARDDataset(
        text=df_test.Text.values,
        target=None,
        config=config,
        is_test=True,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    device = torch.device(config.device)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=2, ignore_mismatched_sizes=True
    )
    model.load_state_dict(
        torch.load(f"{config.checkpoint}/{config.model.split('/')[-1]}.pt")
    )
    model.to(device)
    model.eval()
    fin_outputs = []

    # val_loss, outputs, targets = eval_fn(valid_data_loader, model, device)
    # precision, recall, thresholds = precision_recall_curve(df_valid.Class.values, outputs)
    # fscore = (2 * precision * recall) / (precision + recall)
    # # locate the index of the largest f score
    # ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    with torch.no_grad():
        for bi, inputs in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader)
        ):
            for i in inputs.keys():
                inputs[i] = inputs[i].squeeze(1).to(device)
            outputs = model(**inputs).logits.squeeze(-1)

            fin_outputs.extend(outputs.argmax(axis=1).detach().cpu().numpy().tolist())
    df_test["Class"] = fin_outputs
    # df_test[["Id", "Class"]].to_csv(
    #     f"{config.submission}/prob_submission.csv", index=None
    # )

    # outputs = np.array(df_test["Class"]) >= np.median(fin_outputs) #thresholds[ix]
    # df_test["Class"] = df_test["Class"] >= np.median(fin_outputs)
    df_test["Class"] = df_test["Class"].apply(lambda x: "M" if x else "H")
    df_test[["Id", "Class"]].to_csv(f"{config.submission}/submission.csv", index=None)


if __name__ == "__main__":
    main()
