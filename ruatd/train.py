import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics, model_selection
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2LMHeadModel, get_linear_schedule_with_warmup

from ruatd.dataset import RuARDDataset
from ruatd.engine import eval_fn, train_fn
from ruatd.model import BERTBaseUncased


@hydra.main(config_path="config", config_name="bert")
def main(config: DictConfig):

    wandb.init(
        config=config,
        project=config["project"],
        name=f"{config['classification']}_{config['model']}",
    )
    df_train = pd.read_csv(config.data.train)
    df_valid = pd.read_csv(config.data.val)
    df_test = pd.read_csv(config.data.test)

    df_train.Class = df_train.Class.apply(lambda x: 1 if x == "M" else 0)
    df_valid.Class = df_valid.Class.apply(lambda x: 1 if x == "M" else 0)

    train_dataset = RuARDDataset(
        text=df_train.Text.values, target=df_train.Class.values, config=config
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    valid_dataset = RuARDDataset(
        text=df_valid.Text.values, target=df_valid.Class.values, config=config
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    device = torch.device(config.device)
    model = BERTBaseUncased(config)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config["device_ids"])

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.batch_size * config.epochs)
    optimizer = AdamW(optimizer_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_roc_auc = 0
    for _ in range(config.epochs):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        val_loss, outputs, targets = eval_fn(valid_data_loader, model, device)
        roc_auc = metrics.roc_auc_score(targets, outputs)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if roc_auc > best_roc_auc:
            print("Model saved!")
            torch.save(
                model.module.state_dict(),
                f"{config.checkpoint}/{config.model.split('/')[-1]}.pt",
            )
            best_roc_auc = roc_auc

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": roc_auc,
                "val_accuracy": accuracy,
            }
        )


if __name__ == "__main__":
    main()
