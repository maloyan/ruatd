import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs.view(-1, 1), targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    sum_loss = 0
    for bi, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for i in inputs.keys():
            inputs[i] = inputs[i].squeeze(1).to(device)

        targets = targets.to(device)

        optimizer.zero_grad()
        # import IPython; IPython.embed(); exit(1)
        outputs = model(**inputs).logits.squeeze(-1)

        loss = F.cross_entropy(
            outputs, torch.LongTensor(targets.detach().cpu().numpy()).to(device)
        )
        # loss = loss_fn(outputs, targets)

        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return sum_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    val_loss = 0
    with torch.no_grad():
        for bi, (inputs, targets) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):

            for i in inputs.keys():
                inputs[i] = inputs[i].squeeze(1).to(device)

            targets = targets.to(device)
            outputs = model(**inputs).logits.squeeze(-1)

            val_loss += F.cross_entropy(
                outputs, torch.LongTensor(targets.detach().cpu().numpy()).to(device)
            )
            # val_loss += loss_fn(outputs, targets)
            predictions = outputs.argmax(axis=1)
            fin_targets.extend(targets.detach().cpu().numpy())
            fin_outputs.extend(predictions.detach().cpu().numpy().tolist())
    return val_loss / len(data_loader), fin_outputs, fin_targets
