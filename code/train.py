# python imports
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

# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
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


################################################################################
def main(args):
    """main function that handles training"""

    """1. Load config / Setup folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg["output_folder"]):
        os.mkdir(cfg["output_folder"])
    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg["output_folder"], cfg_filename + "_" + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg["output_folder"], cfg_filename + "_" + str(args.output)
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    """2. create dataset / dataloader"""
    train_dataset = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["train"],
        cfg["dataset"]["img_folder"],
        cfg["dataset"]["json_folder"],
    )
    # data loaders
    train_loader = build_dataloader(train_dataset, True, **cfg["loader"])

    """3. create model, optimizer, and scheduler"""
    # model
    model = FCOS(**cfg["model"]).to(torch.device(cfg["devices"][0]))
    # optimizer
    optimizer = build_optimizer(model, cfg["opt"])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)
    # also disable cudnn benchmark, as the input size varies during training
    cudnn.benchmark = False

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(cfg["devices"][0]),
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{:s}' (epoch {:d}".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, "config.txt"), "w") as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training loop"""
    print("\nStart training ...")

    # start training
    max_epochs = cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"]
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            device=torch.device(cfg["devices"][0]),
            tb_writer=tb_writer,
            print_freq=args.print_freq,
        )

        # save ckpt once in a while
        if ((epoch + 1) == max_epochs) or (
            (args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)
        ):
            save_states = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            is_final = (epoch + 1) == max_epochs
            save_checkpoint(
                save_states,
                is_final,
                file_folder=ckpt_folder,
                file_name="epoch_{:03d}.pth.tar".format(epoch),
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return


################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train a point-based transformer for action localization"
    )
    parser.add_argument("config", metavar="DIR", help="path to a config file")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency (default: 10 iterations)",
    )
    parser.add_argument(
        "-c",
        "--ckpt-freq",
        default=1,
        type=int,
        help="checkpoint frequency (default: every 5 epochs)",
    )
    parser.add_argument(
        "--output", default="", type=str, help="name of exp folder (default: none)"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to a checkpoint (default: none)",
    )
    args = parser.parse_args()
    main(args)
