import sys

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from ruatd.dataset import RuARDDataset
from ruatd.engine import predict_fn


@hydra.main(config_path="config", config_name="binary")
def main(config: DictConfig):
    df_valid = pd.read_csv(config.data.val)
    df_test = pd.read_csv(config.data.test)

    valid_dataset = RuARDDataset(
        text=df_valid.Text.values, target=None, config=config, is_test=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

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
        config.model, num_labels=config.num_classes, ignore_mismatched_sizes=True
    )
    model.load_state_dict(
        torch.load(
            f"{config.checkpoint}/{config.classification}_{config.model.split('/')[-1]}.pt"
        )
    )
    model.to(device)
    model.eval()

    prob_valid, df_valid["Class"] = predict_fn(valid_data_loader, model, config)
    prob_test, df_test["Class"] = predict_fn(test_data_loader, model, config)

    if config.classification == "multiclass":
        class_dict = {
            0: "ruGPT3-Small",
            1: "ruGPT3-Medium",
            2: "OPUS-MT",
            3: "M2M-100",
            4: "ruT5-Base-Multitask",
            5: "Human",
            6: "M-BART50",
            7: "ruGPT3-Large",
            8: "ruGPT2-Large",
            9: "M-BART",
            10: "ruT5-Large",
            11: "ruT5-Base",
            12: "mT5-Large",
            13: "mT5-Small",
        }
    else:
        class_dict = {0: "H", 1: "M"}
    pd.concat(
        [df_valid["Id"], pd.DataFrame(prob_valid).rename(columns=class_dict)], axis=1
    ).to_csv(
        f"{config.submission}/{config.classification}/prob_valid_{config.model.split('/')[-1]}.csv",
        index=None,
    )

    pd.concat(
        [df_test["Id"], pd.DataFrame(prob_test).rename(columns=class_dict)], axis=1
    ).to_csv(
        f"{config.submission}/{config.classification}/prob_test_{config.model.split('/')[-1]}.csv",
        index=None,
    )

    df_test.Class = df_test.Class.map(class_dict)
    df_test[["Id", "Class"]].to_csv(
        f"{config.submission}/{config.classification}/class_{config.model.split('/')[-1]}.csv",
        index=None,
    )


if __name__ == "__main__":
    main()
