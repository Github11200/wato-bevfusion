"""Single-GPU training script for BEVFusion segmentation (no DDP overhead)."""
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.runner.hooks import CheckpointHook, TextLoggerHook
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--run-dir", default="runs/seg-resnet50")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    os.makedirs(args.run_dir, exist_ok=True)
    cfg.run_dir = args.run_dir

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True,
    )

    model = build_model(cfg.model)
    model.init_weights()
    model = MMDataParallel(model.cuda(), device_ids=[0])

    optimizer = build_optimizer(model, cfg.optimizer)

    logger = logging.getLogger("mmdet3d")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=args.run_dir,
        logger=logger,
        max_epochs=cfg.runner.max_epochs,
    )

    runner.register_hook(CheckpointHook(interval=1, max_keep_ckpts=1))
    runner.register_hook(TextLoggerHook(interval=50))
    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.get("checkpoint_config", dict(interval=1)),
        cfg.log_config,
        cfg.get("momentum_config", None),
    )

    runner.run([dataloader], [("train", 1)])


if __name__ == "__main__":
    main()
