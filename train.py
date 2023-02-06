from collections import OrderedDict

import pandas as pd
import torch
from tqdm import tqdm

from dataloader import get_loaders
from losses import BCEDiceLoss
from metrics import iou_score
from models import NestedUNet
from utils import AverageMeter


def train(train_loader, model, criterion, optimizer):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters["loss"].update(loss.item(), input.shape[0])
        avg_meters["iou"].update(iou, input.shape[0])

        postfix = OrderedDict(
            [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return postfix


def validate(val_loader, model, criterion):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}
    model.eval()
    pbar = tqdm(total=len(val_loader))
    for input, target in val_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)

        avg_meters["loss"].update(loss.item(), input.shape[0])
        avg_meters["iou"].update(iou, input.shape[0])

        postfix = OrderedDict(
            [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return postfix


def initialize():
    model = NestedUNet(1).cuda()
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return model, criterion, optimizer


if __name__ == "__main__":
    model, criterion, optimizer = initialize()

    train_loader, val_loader = get_loaders()

    log = OrderedDict(
        [
            ("epoch", []),
            ("loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_iou", []),
        ]
    )

    best_iou = 0
    trigger = 0

    for epoch in range(50):
        print(f"Epoch [{epoch}/50]")

        train_log = train(train_loader, model, criterion, optimizer)
        val_log = validate(val_loader, model, criterion)
        print(
            "loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f"
            % (
                train_log["loss"],
                train_log["iou"],
                val_log["loss"],
                val_log["iou"],
            )
        )
        log["epoch"].append(epoch)
        log["loss"].append(train_log["loss"])
        log["iou"].append(train_log["iou"])
        log["val_loss"].append(val_log["loss"])
        log["val_iou"].append(val_log["iou"])

        pd.DataFrame(log).to_csv("models/log.csv", index=False)

        trigger += 1

        if val_log["iou"] > best_iou:
            torch.save(model.state_dict(), "models/model.pt")
            best_iou = val_log["iou"]
            print("=> saved best model")
            trigger = 0

        if trigger >= 5:
            print("=> early stopping")
            break
