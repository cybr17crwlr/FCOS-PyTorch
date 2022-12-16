import os
import random

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .transforms import Compose, ConvertAnnotations, RandomHorizontalFlip, ToTensor


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))


def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class VOCDetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load VOC data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def get_voc_cls_names(self):
        cls_names = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        return cls_name

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_dataset(name, split, img_folder, json_folder):
    """
    Create VOC dataset with default transforms for training / inference
    """
    assert name == "VOC2007"
    assert split in ["trainval", "test"]

    is_training = split == "trainval"
    if is_training:
        transforms = Compose([ConvertAnnotations(), RandomHorizontalFlip(), ToTensor()])
    else:
        transforms = Compose([ConvertAnnotations(), ToTensor()])

    dataset = VOCDetection(
        img_folder, os.path.join(json_folder, split + ".json"), transforms
    )
    return dataset


def build_dataloader(dataset, is_training, batch_size, num_workers):
    """
    Create a dataloder for VOC dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=True,
    )
    return loader
