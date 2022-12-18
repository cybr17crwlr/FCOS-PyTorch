import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data


from libs import (
    load_config,
    build_dataset,
    build_dataloader,
    FCOS,
    train_one_epoch,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)

def main():
    cfg = load_config("configs/voc_fcos.yaml")
    pprint(cfg)
    train_dataset = build_dataset(
            cfg["dataset"]["name"],
            cfg["dataset"]["train"],
            cfg["dataset"]["img_folder"],
            cfg["dataset"]["json_folder"],
        )
        # data loaders
    train_loader = build_dataloader(train_dataset, True, **cfg["loader"])

    len(train_loader)

    model = FCOS(**cfg["model"]).to(torch.device(cfg["devices"][0]))

    optimizer = build_optimizer(model, cfg["opt"])
        # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)

    max_epochs = cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"]
    for epoch in range(0, max_epochs):
            # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            device=torch.device(cfg["devices"][0]),
        )


if __name__ == '__main__':
    main()
