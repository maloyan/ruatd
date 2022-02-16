import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

from ruatd.dataset import RuARDDataset
from ruatd.engine import eval_fn, train_fn, predict_fn


def run_fold(config, fold_num, df_train, df_valid, df_test):
    wandb.init(
        config=config,
        project=config["project"],
        name=f"{config['classification']}_{config['model']}_fold{fold_num}",
    )
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
                f"{config.checkpoint}/{config.classification}/{config.model.split('/')[-1]}_fold{fold_num}.pt",
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

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=config.num_classes, ignore_mismatched_sizes=True
    )
    model.load_state_dict(
        torch.load(
            f"{config.checkpoint}/{config.classification}/{config.model.split('/')[-1]}_fold{fold_num}.pt",
        )
    )
    model.to(device)
    model.eval()
    valid_dataset = RuARDDataset(
        text=df_valid.Text.values, target=None, config=config, is_test=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

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
        [df_valid["Id"].reset_index(), pd.DataFrame(prob_valid).rename(columns=class_dict)], axis=1
    ).to_csv(
        f"{config.submission}/{config.classification}/prob_valid_{config.model.split('/')[-1]}_fold{fold_num}.csv",
        index=None,
    )
    pd.concat(
        [df_test["Id"], pd.DataFrame(prob_test).rename(columns=class_dict)], axis=1
    ).to_csv(
        f"{config.submission}/{config.classification}/prob_test_{config.model.split('/')[-1]}_fold{fold_num}.csv",
        index=None,
    )

    df_test.Class = df_test.Class.map(class_dict)
    df_test[["Id", "Class"]].to_csv(
        f"{config.submission}/{config.classification}/class_{config.model.split('/')[-1]}_fold{fold_num}.csv",
        index=None,
    )
    wandb.finish()
@hydra.main(config_path="config", config_name="binary")
def main(config: DictConfig):

    df_train = pd.read_csv(config.data.train)
    df_valid = pd.read_csv(config.data.val)

    df_train = pd.concat([df_train, df_valid])
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
        # df_multi_train = pd.read_csv(config.data.train.replace("binary", "multiclass"))
        # df_multi_train.Class = df_multi_train.Class.apply(
        #     lambda x: 0 if x == "Human" else 1
        # )

        # df_multi_val = pd.read_csv(config.data.val.replace("binary", "multiclass"))
        # df_multi_val.Class = df_multi_val.Class.apply(
        #     lambda x: 0 if x == "Human" else 1
        # )
        class_dict = {"H": 0, "M": 1}
    print(class_dict)
    df_train.Class = df_train.Class.map(class_dict)
    # df_valid.Class = df_valid.Class.map(class_dict)
    # if config.classification == "binary":
    #     df_train = pd.concat([df_train, df_multi_val])
    #     df_train.Class = df_train.Class.astype(int)
    print(df_train.head())
    # print(df_valid.head())
    skf = StratifiedKFold(n_splits=config.num_folds)
    skf.get_n_splits(df_train)
    for fold_num, (train_index, test_index) in enumerate(skf.split(df_train, df_train["Class"])):
        run_fold(config, fold_num, df_train.iloc[train_index], df_train.iloc[test_index], df_test)


if __name__ == "__main__":
    main()
