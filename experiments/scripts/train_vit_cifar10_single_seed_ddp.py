#!/usr/bin/env python3
"""
Train ONE ViT-B/16 seed on CIFAR-10 with DDP, and save probe features per epoch.

- Uses torchrun/DDP
- Saves:
    out_dir/probe_indices.npy  (shared across seeds)
    out_dir/features/seed_{seed}/epoch_{e}/block{b}_{cls|mean}.npy
    out_dir/checkpoints/seed_{seed}/epoch_{e}.pth
- Also saves epoch_0 features at random init.

Run example (2 GPUs):
torchrun --nproc_per_node=2 train_vit_cifar10_single_seed_ddp.py \
  --seed 0 --out-dir /p/scratch/.../vit_c10_conv --epochs 100
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

import torchvision
import torchvision.transforms as T
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler


# --------------------------- DDP setup ---------------------------

def ddp_setup():
    import os
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # <-- critical: bind rank to GPU

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=local_rank,           # <-- tells NCCL the device for this rank
    )
    dist.barrier(device_ids=[local_rank])  # <-- optional but silences warning
    return local_rank
def ddp_cleanup():
    dist.destroy_process_group()

def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()


# --------------------------- seeding ---------------------------

def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --------------------------- transforms ---------------------------
#****************To be confirmed from HuggingFace*******************
#****************Training method from Huggingface ******************

def cifar10_train_transforms(img_size=224):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std  = (0.24703233, 0.24348505, 0.26158768)
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(img_size, padding=img_size // 8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])



def cifar10_test_transforms(img_size=224):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std  = (0.24703233, 0.24348505, 0.26158768)
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

#*********************************************************************

# --------------------------- probe ---------------------------

def build_class_balanced_probe(targets: np.ndarray, n_total: int, seed: int = 0, n_classes: int = 10):
    rng = np.random.default_rng(seed)
    per_class = n_total // n_classes
    idxs = []
    for c in range(n_classes):
        c_idx = np.where(targets == c)[0]
        rng.shuffle(c_idx)
        idxs.append(c_idx[:per_class])
    idxs = np.concatenate(idxs)
    rng.shuffle(idxs)
    return idxs


# --------------------------- model ---------------------------

def make_vit(num_classes=10):
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )


# --------------------------- training ---------------------------

def train_one_epoch(model, loader, optimizer, device, loss_fn, grad_clip=1.0):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_accuracy(model, loader, device, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


# --------------------------- feature extraction (rank0 only) ---------------------------


@torch.no_grad()
def extract_vit_block_features(model, loader, blocks, device, save_fp16=True):
    model.eval()
    vit = model.module if isinstance(model, DDP) else model
    feats = {(b, "cls"): [] for b in blocks}
    feats.update({(b, "mean"): [] for b in blocks})

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        x = vit.patch_embed(images)
        cls_tokens = vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)

        for i, blk in enumerate(vit.blocks, start=1):
            x = blk(x)
            if i in blocks:
                x_norm = vit.norm(x)
                cls = x_norm[:, 0]
                mean = x_norm[:, 1:].mean(1)
                if save_fp16:
                    cls = cls.half()
                    mean = mean.half()
                feats[(i, "cls")].append(cls.cpu())
                feats[(i, "mean")].append(mean.cpu())

    return {k: torch.cat(v, 0).numpy() for k, v in feats.items()}


def save_epoch_features(feats, feat_root: Path):
    feat_root.mkdir(parents=True, exist_ok=True)
    for (b, pooling), arr in feats.items():
        np.save(feat_root / f"block{b}_{pooling}.npy", arr)


# --------------------------- main ---------------------------

def main():
    import os
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--probe-size", type=int, default=10000)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--blocks", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--save-fp16", action="store_true")

    parser.add_argument("--out-dir", type=str, required=True)

    args = parser.parse_args()

    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    feat_dir = out_dir / "features" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Datasets / loaders
    train_ds = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=False,
        transform=cifar10_train_transforms()
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=False,
        transform=cifar10_test_transforms()
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Fixed probe indices shared across seeds
    targets = np.array(test_ds.targets)
    probe_path = out_dir / "probe_indices.npy"

    if is_rank0():
        if probe_path.exists():
            probe_indices = np.load(probe_path)
        else:
            probe_indices = build_class_balanced_probe(
                targets, args.probe_size, seed=args.probe_seed
            )
            np.save(probe_path, probe_indices)
        print(f"[Rank0] probe size={len(probe_indices)} saved/loaded at {probe_path}")
    barrier()

    probe_indices = np.load(probe_path)
    probe_ds = Subset(test_ds, probe_indices)
    probe_loader = DataLoader(
        probe_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model / optimizer / scheduler
    model = make_vit().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.lr * 0.1
    )

    loss_fn = nn.CrossEntropyLoss()

    # -------- Epoch 0 features (random init baseline) --------
    if is_rank0():
        print("[Rank0] Extracting epoch_0 features (random init)...")
        feats0 = extract_vit_block_features(model, probe_loader, args.blocks, device, save_fp16=args.save_fp16)
        save_epoch_features(feats0, feat_dir / "epoch_0")

    # -------- Train epochs --------
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn
        )
        scheduler.step(epoch)

        if is_rank0():
            te_loss, te_acc = eval_accuracy(model, test_loader, device, loss_fn)
            print(f"[Seed {args.seed}] epoch={epoch} train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")

            # save ckpt
            torch.save(
                model.module.state_dict(),
                ckpt_dir / f"seed_{args.seed}_epoch_{epoch}.pth"
            )

            # extract probe features
            feats = extract_vit_block_features(model, probe_loader, args.blocks, device, save_fp16=args.save_fp16)
            save_epoch_features(feats, feat_dir / f"epoch_{epoch}")

        barrier()

    if is_rank0():
        print(f"[Seed {args.seed}] Training done.")
    ddp_cleanup()


if __name__ == "__main__":
    main()