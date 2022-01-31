import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from ruatd.dataset import RuARDDataset
from ruatd.engine import eval_fn, train_fn


@hydra.main(config_path="config", config_name="binary")
def main(config: DictConfig):

    wandb.init(
        config=config,
        project=config["project"],
        name=f"{config['classification']}_{config['model']}",
    )
    df_train = pd.read_csv(config.data.train)
    df_valid = pd.read_csv(config.data.val)
    df_test = pd.read_csv(config.data.test)
    if config.classification == "multiclass":
        class_dict = {
            "ruGPT3-Small": 0,
            "ruGPT3-Medium": 1,
            "OPUS-MT": 2,
            "M2M-100": 3,
            "ruT5-Base-Multitask": 4,
            "Human": 5,
            "M-BART50": 6,
            "ruGPT3-Large": 7,
            "ruGPT2-Large": 8,
            "M-BART": 9,
            "ruT5-Large": 10,
            "ruT5-Base": 11,
            "mT5-Large": 12,
            "mT5-Small": 13,
        }
    else:
        class_dict = {
            "H": 0,
            "M": 1
        }
    print(class_dict)
    df_train.Class = df_train.Class.map(class_dict)
    df_valid.Class = df_valid.Class.map(class_dict)
    print(df_train.head())
    print(df_valid.head())
    train_dataset = RuARDDataset(
        text=df_train.Text.values, target=df_train.Class.values, config=config
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    valid_dataset = RuARDDataset(
        text=df_valid.Text.values, target=df_valid.Class.values, config=config
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    device = torch.device(config.device)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=config.num_classes, ignore_mismatched_sizes=True
    )
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

    best_loss = 10000
    for _ in range(config.epochs):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        val_loss, outputs, targets = eval_fn(valid_data_loader, model, device)
        # roc_auc = metrics.roc_auc_score(targets, outputs)
        # outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if val_loss < best_loss:
            print("Model saved!")
            torch.save(
                model.module.state_dict(),
                f"{config.checkpoint}/{config.classification}_{config.model.split('/')[-1]}.pt",
            )
            best_loss = val_loss

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                # "val_roc_auc": roc_auc,
                "val_accuracy": accuracy,
            }
        )


if __name__ == "__main__":
    main()
