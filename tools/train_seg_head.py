"""Train neck + vtransform + seg head using cached lidar features.

Two-phase training to fit in 8GB VRAM:
  Phase 1 (cache_lidar_features.py): precompute lidar BEV features on GPU
  Phase 2 (this script): train camera pipeline + fuser + decoder + seg head,
    reading cached lidar features from disk instead of running SparseEncoder.

Frozen: camera backbone (ResNet50), fuser, decoder
Trainable: camera neck, camera vtransform, seg head
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def load_partial_checkpoint(model, ckpt_path):
    """Load fuser and decoder weights from SwinT checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    loadable_prefixes = (
        "fuser.",
        "decoder.",
    )

    model_sd = model.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if any(k.startswith(p) for p in loadable_prefixes):
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

    model.load_state_dict(model_sd)
    print(f"Loaded {len(loaded)} params from checkpoint (fuser, decoder)")
    if skipped:
        print(f"Skipped {len(skipped)} params (shape mismatch): {skipped[:5]}")


def freeze_module(module, name=""):
    count = 0
    for param in module.parameters():
        param.requires_grad = False
        count += 1
    if count:
        print(f"  Froze {name}: {count} parameters")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--pretrained", default="pretrained/bevfusion-seg.pth")
    parser.add_argument("--lidar-cache", default="cached_lidar_features")
    parser.add_argument("--run-dir", default="runs/seg-resnet50")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only train on first N samples per epoch (for testing)")
    parser.add_argument("--lr", type=float, default=2e-4)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    # Disable 3D augmentation to match cached lidar features
    configs.update([
        "--augment3d.scale", "[1.0, 1.0]",
        "--augment3d.rotate", "[0.0, 0.0]",
        "--augment3d.translate", "0.0",
    ])
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    os.makedirs(args.run_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    print("Building model...")
    model = build_model(cfg.model)
    model.init_weights()

    print(f"Loading fuser + decoder from {args.pretrained}...")
    load_partial_checkpoint(model, args.pretrained)

    # Remove lidar encoder to save GPU memory — cached features replace it
    if "lidar" in model.encoders:
        del model.encoders["lidar"]
        print("  Removed lidar encoder from model (using cached features)")

    print("Freezing layers...")
    freeze_module(model.encoders["camera"]["backbone"], "camera backbone (ResNet50)")
    freeze_module(model.fuser, "fuser")
    freeze_module(model.decoder, "decoder")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters "
          f"({100*trainable/total:.1f}%)")

    # Use inner dataset (skip CBGS wrapper) for stable index mapping
    dataset = build_dataset(cfg.data.train)
    inner_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    dataloader = build_dataloader(
        inner_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
        dist=False,
        shuffle=False,
    )

    # Calculate how many samples we'll actually use (respecting max_samples flag)
    n_dataset = min(args.max_samples, len(inner_dataset)) if args.max_samples else len(inner_dataset)

    n_cached = len([f for f in os.listdir(args.lidar_cache) if f.endswith(".pt")])
    print(f"Found {n_cached} cached lidar features in {args.lidar_cache}/")
    if n_cached < n_dataset:
        print(f"WARNING: Only {n_cached} cached features available but planning to train on {n_dataset} samples")
        print(f"Reducing training samples to {n_cached}")
        n_dataset = n_cached

    model = model.cuda()
    print(f"GPU memory after model.cuda(): "
          f"{torch.cuda.memory_allocated()/1024**2:.0f} MB")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)

    print(f"\nStarting training for {args.max_epochs} epochs...")
    print(f"  Training samples: {n_dataset} (total dataset: {len(inner_dataset)}, cached: {n_cached})")
    print(f"  Learning rate: {args.lr}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB / 8192MB")

    model.train()
    model.encoders["camera"]["backbone"].eval()
    model.fuser.eval()
    model.decoder.eval()

    for epoch in range(args.max_epochs):
        epoch_loss = 0.0
        epoch_losses = {}

        # Only use indices that have cached lidar features
        # Shuffle order each epoch for training variance
        order = list(range(min(n_cached, n_dataset)))
        random.shuffle(order)

        for step, sample_idx in enumerate(order):
            if args.max_samples and step >= args.max_samples:
                break

            # Get data from dataloader by index
            data = inner_dataset[sample_idx]
            data = collate([data], samples_per_gpu=1)

            # Load cached lidar features
            lidar_bev = torch.load(
                os.path.join(args.lidar_cache, f"{sample_idx:06d}.pt"),
                map_location="cuda",
            )

            img = data["img"].data[0].cuda()
            points = data["points"].data[0]
            points = [p.cuda() for p in points]
            metas = data["metas"].data[0]

            camera2ego = data["camera2ego"].data[0].cuda()
            lidar2ego = data["lidar2ego"].data[0].cuda()
            lidar2camera = data["lidar2camera"].data[0].cuda()
            lidar2image = data["lidar2image"].data[0].cuda()
            camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
            camera2lidar = data["camera2lidar"].data[0].cuda()
            img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
            lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()
            gt_masks_bev = data["gt_masks_bev"].data.cuda()  # Keep batch dimension

            # Camera backbone (frozen, no grad needed)
            with torch.no_grad():
                B, N, C, H, W = img.size()
                x = img.view(B * N, C, H, W)
                x = model.encoders["camera"]["backbone"](x)

            # Neck + vtransform (trainable — gradients flow)
            x = model.encoders["camera"]["neck"](x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            BN, C, H, W = x.size()
            x = x.view(B, int(BN / B), C, H, W)

            camera_bev = model.encoders["camera"]["vtransform"](
                x, points,
                camera2ego, lidar2ego, lidar2camera, lidar2image,
                camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix,
                metas=metas,
            )

            # Fuser + decoder + seg head
            # Params frozen but graph must connect for backprop to neck/vtransform
            fused = model.fuser([camera_bev, lidar_bev])
            x = model.decoder["backbone"](fused)
            x = model.decoder["neck"](x)

            # Handle case where neck returns a list (multiple scales)
            if isinstance(x, list):
                x = x[0] if len(x) == 1 else torch.cat(x, dim=1)

            losses = model.heads["map"](x, gt_masks_bev)

            loss = sum(v for v in losses.values() if v.requires_grad)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=35)
            optimizer.step()

            epoch_loss += loss.item()
            for loss_name, loss_val in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = 0.0
                epoch_losses[loss_name] += loss_val.item()

            # Log every 5 samples or at start
            log_interval = max(1, n_dataset // 10) if n_dataset > 10 else 1
            if (step + 1) % log_interval == 0 or step == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**2
                avg_epoch_loss = epoch_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{args.max_epochs}] "
                      f"Step [{step+1}/{n_dataset}] "
                      f"loss={loss.item():.4f} "
                      f"avg_loss={avg_epoch_loss:.4f} "
                      f"peak_mem={mem:.0f}MB")

            del img, lidar_bev, camera_bev, fused, x, losses, loss, data
            torch.cuda.empty_cache()

        # End of epoch
        avg_epoch_loss = epoch_loss / max(step + 1, 1)
        loss_str = " ".join([f"{k}={v/(step+1):.4f}" for k, v in epoch_losses.items()])
        print(f"Epoch {epoch+1}/{args.max_epochs} done | avg_loss={avg_epoch_loss:.4f} | {loss_str}")

        ckpt_path = os.path.join(args.run_dir, "checkpoint.pth")
        torch.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}\n")

    # Save a clean weights-only file for sharing (no optimizer state)
    latest_path = os.path.join(args.run_dir, "latest.pth")
    torch.save({"state_dict": model.state_dict()}, latest_path)
    print(f"Saved shareable weights: {latest_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
