"""Phase 1: Cache lidar BEV features to disk.

Builds ONLY the lidar encoder (not the full BEVFusion model) to
minimize GPU memory. Runs through every training sample and saves
the dense BEV feature tensor.

3D augmentation is disabled so features are deterministic per sample.
"""
import argparse
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models.builder import build_backbone
from mmdet3d.ops import Voxelization
from mmdet3d.utils import recursive_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--pretrained", default="pretrained/bevfusion-seg.pth")
    parser.add_argument("--out-dir", default="cached_lidar_features")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only cache first N samples (for testing)")
    args = parser.parse_args()

    configs.load(args.config, recursive=True)
    # Disable 3D augmentation for deterministic features
    configs.update([
        "--augment3d.scale", "[1.0, 1.0]",
        "--augment3d.rotate", "[0.0, 0.0]",
        "--augment3d.translate", "0.0",
        "--augment2d.resize", "[[0.48, 0.48], [0.48, 0.48]]",
        "--augment2d.rotate", "[0.0, 0.0]",
    ])
    cfg = Config(recursive_eval(configs), filename=args.config)

    os.makedirs(args.out_dir, exist_ok=True)

    lidar_cfg = cfg.model.encoders.lidar
    print("Building lidar encoder only (not full model)...")

    voxelize = Voxelization(**lidar_cfg.voxelize)
    backbone = build_backbone(lidar_cfg.backbone)
    voxelize_reduce = lidar_cfg.get("voxelize_reduce", True)

    # Load lidar backbone weights from pretrained checkpoint
    ckpt = torch.load(args.pretrained, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    backbone_sd = {}
    prefix = "encoders.lidar.backbone."
    for k, v in sd.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            backbone_sd[new_k] = v
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    print(f"Loaded {len(backbone_sd)} lidar backbone params "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

    backbone = backbone.cuda()
    backbone.eval()
    print(f"GPU memory after lidar backbone.cuda(): "
          f"{torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # Build dataset without CBGS wrapper for consistent indexing
    dataset = build_dataset(cfg.data.train)
    inner_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    dataloader = build_dataloader(
        inner_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    print(f"Dataset has {len(inner_dataset)} samples")
    print(f"Caching lidar features to {args.out_dir}...")

    n_samples = min(args.max_samples, len(inner_dataset)) if args.max_samples else len(inner_dataset)
    print(f"Will cache {n_samples} samples (max_samples={args.max_samples})")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= n_samples:
                break

            points = data["points"].data[0]
            points_gpu = [p.cuda() for p in points]

            # Voxelize (runs on GPU via custom CUDA op)
            feats_list, coords_list, sizes_list = [], [], []
            for k, res in enumerate(points_gpu):
                ret = voxelize(res)
                if len(ret) == 3:
                    f, c, n = ret
                else:
                    f, c = ret
                    n = None
                feats_list.append(f)
                coords_list.append(F.pad(c, (1, 0), mode="constant", value=k))
                if n is not None:
                    sizes_list.append(n)

            feats = torch.cat(feats_list, dim=0)
            coords = torch.cat(coords_list, dim=0)
            if len(sizes_list) > 0:
                sizes = torch.cat(sizes_list, dim=0)
                if voxelize_reduce:
                    feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                    feats = feats.contiguous()
            else:
                sizes = torch.zeros(0, dtype=torch.long, device=feats.device)

            batch_size = coords[-1, 0] + 1
            lidar_bev = backbone(feats, coords, batch_size, sizes=sizes)

            save_path = os.path.join(args.out_dir, f"{i:06d}.pt")
            torch.save(lidar_bev.cpu(), save_path)

            if (i + 1) % max(1, n_samples // 5) == 0 or i == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  [{i+1}/{n_samples}] shape={lidar_bev.shape} "
                      f"peak_mem={mem:.0f}MB saved to {save_path}")

            del feats, coords, sizes, lidar_bev, points_gpu, feats_list, coords_list, sizes_list
            torch.cuda.empty_cache()

    print(f"Done. Cached {len(inner_dataset)} lidar feature tensors in {args.out_dir}/")


if __name__ == "__main__":
    main()
