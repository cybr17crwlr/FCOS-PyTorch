import yaml

# DEFAULT defines the default params used for training / inference
# the parameters here will be overwritten if a yaml config is specified
DEFAULTS = {
    # default: single gpu
    "devices": ["cuda:0"],
    "model_name": "FCOS",
    # output folder that stores all log files and checkpoints
    "output_folder": None,
    "dataset": {
        "name": "VOC2007",
        # training / testing splits
        "train": "trainval",
        "test": "test",
        # folders that store image files / json annotations (following COCO format)
        "img_folder": None,
        "json_folder": None,
    },
    "loader": {
        "batch_size": 4,
        "num_workers": 4,
    },
    "input": {
        # min / max size of the input image
        "img_min_size": [256, 288, 320],
        "img_max_size": 480,
        # mean / std for input normalization
        "img_mean": [0.485, 0.456, 0.406],
        "img_std": [0.229, 0.224, 0.225],
    },
    # network architecture
    "model": {
        # type of backbone network
        "backbone": "ResNet18",
        # output features from backbone (also as the input fetures for FPN)
        "backbone_out_feats": ["layer2", "layer3", "layer4"],
        # output feature dimensions
        "backbone_out_feats_dims": [128, 256, 512],
        # feature dimension after FPN
        "fpn_feats_dim": 128,
        # feature strides in FPN
        "fpn_strides": [8, 16, 32],
        # number of object categories (excluding background)
        "num_classes": 20,
        # regression range for each pyramid level
        "regression_range": [(0, 32), (32, 64), (64, 128)],
    },
    "train_cfg": {
        # radius used for center sampling during training
        "center_sampling_radius": 1.5,
        # additional params can be added here
    },
    "test_cfg": {
        # score threshold for postprocessing
        "score_thresh": 0.1,
        # NMS threshold for postprocessing
        "nms_thresh": 0.6,
        # maximum number of boxes per image (after NMS)
        "detections_per_img": 100,
        # maximum number of boxes per image (before NMS)
        "topk_candidates": 1000,
        # additional params can be added here
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "SGD",
        # solver params
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "learning_rate": 5e-3,
        # excluding the warmup epochs
        "epochs": 10,
        # if to use linear warmup
        "warmup": True,
        "warmup_epochs": 1,
        # lr scheduler: cosine / multistep
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    config = DEFAULTS
    return config


def _update_config(config):
    config["model"].update(config["input"])
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config
