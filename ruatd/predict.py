import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics, model_selection
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, GPT2LMHeadModel, get_linear_schedule_with_warmup

from ruatd.dataset import RuARDDataset
from ruatd.engine import eval_fn, train_fn
from ruatd.model import BERTBaseUncased


@hydra.main(config_path="config", config_name="bert")
def main(config: DictConfig):
    df_test = pd.read_csv(config.data.test)

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
    model = BERTBaseUncased(config)
    model.load_state_dict(
        torch.load(f"{config.checkpoint}/{config.model.split('/')[-1]}.pt")
    )
    model.to(device)
    model.eval()
    fin_outputs = []

    with torch.no_grad():
        for bi, inputs in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader)
        ):
            for i in inputs.keys():
                inputs[i] = inputs[i].to(device)
            outputs = model(**inputs).squeeze(-1)

            fin_outputs.extend(outputs.detach().cpu().numpy().tolist())

    outputs = np.array(fin_outputs) >= 0.5
    df_test["Class"] = df_test["Class"].apply(lambda x: "M" if x else "H")
    df_test[["Id", "Class"]].to_csv("submission.csv", index=None)


if __name__ == "__main__":
    main()
