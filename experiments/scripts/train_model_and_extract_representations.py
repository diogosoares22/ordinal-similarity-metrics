#!/usr/bin/env python3
"""
Script to train a single model and extract representations across epochs.

At each epoch, saves the representations from the test set.
Optionally extracts representations from a corrupted dataset (e.g., CIFAR-10-C).

Based on code from:
https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
import sys

from models.initialization import initialize_weights
from models.vit import ViT_CIFAR10, ViT_CIFAR100
from models.resnet import ResNet50_CIFAR10, ResNet50_CIFAR100


def compute_pairwise_euclidean_distance_stats(representations, n_samples=10000, seed=42):
    """
    Compute min, max, and average pairwise Euclidean distance for a set of representations.
    
    Uses sampling to efficiently estimate statistics for large datasets.
    For n points, there are n*(n-1)/2 unique pairs. We sample n_samples pairs
    to estimate the statistics.
    
    Args:
        representations: numpy array of shape (N, D) where N is number of points
        n_samples: number of random pairs to sample for estimation
        seed: random seed for reproducibility
    
    Returns:
        dict with keys 'min', 'max', 'avg' containing the respective statistics
    """
    n_points = representations.shape[0]
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)
    
    # For small datasets, compute exact statistics
    max_exact_pairs = n_samples
    total_pairs = n_points * (n_points - 1) // 2
    
    if total_pairs <= max_exact_pairs:
        # Compute all pairwise distances exactly using broadcasting
        # This is memory efficient for smaller datasets
        distances = []
        for i in range(n_points):
            # Compute distance from point i to all points j > i
            diffs = representations[i+1:] - representations[i]
            dists = np.linalg.norm(diffs, axis=1)
            distances.extend(dists)
        distances = np.array(distances)
        return {'min': np.min(distances), 'max': np.max(distances), 'avg': np.mean(distances)}
    
    # For large datasets, sample random pairs
    # Generate random indices for pairs
    idx1 = rng.randint(0, n_points, size=n_samples)
    idx2 = rng.randint(0, n_points, size=n_samples)
    
    # Ensure we don't sample the same point twice (i != j)
    same_idx = idx1 == idx2
    while np.any(same_idx):
        idx2[same_idx] = rng.randint(0, n_points, size=np.sum(same_idx))
        same_idx = idx1 == idx2
    
    # Compute distances for sampled pairs
    diffs = representations[idx1] - representations[idx2]
    distances = np.linalg.norm(diffs, axis=1)
    
    return {'min': np.min(distances), 'max': np.max(distances), 'avg': np.mean(distances)}


def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train model and extract representations across epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--net', default='vit', choices=['vit', 'res50'], help='network architecture')
    parser.add_argument('--bs', default=512)
    parser.add_argument('--size', default=32)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--custom-initialization', action='store_true', help='use custom initialization')
    parser.add_argument('--init-gain', default=1, type=float, help='gain for weight initialization')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate for exponential decay')
    parser.add_argument('--save-representations', action='store_true', default=True, help='save representations to disk (can be large)')
    parser.add_argument('--corrupted-data-path', type=str, default=None, 
                        help='path to corrupted dataset .npy file (e.g., CIFAR-10-C gaussian noise)')
    parser.add_argument('--extract-only', action='store_true', 
                        help='skip training and only extract representations from a saved model')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='path to saved model checkpoint (used with --extract-only)')
    return parser.parse_args()


class CorruptedCIFAR10Dataset(torch.utils.data.Dataset):
    """
    Dataset for corrupted CIFAR-10 data loaded from .npy file.
    Applies min-max normalization followed by standard CIFAR-10 normalization.
    """
    def __init__(self, npy_path, size=32):
        """
        Args:
            npy_path: Path to .npy file with shape (N, 32, 32, 3) and values in [0, 255]
            size: Target image size (default 32)
        """
        print(f"Loading corrupted data from: {npy_path}")
        self.data = np.load(npy_path)
        print(f"Loaded data shape: {self.data.shape}, dtype: {self.data.dtype}")
        print(f"Data range: [{self.data.min()}, {self.data.max()}]")
        
        # CIFAR-10 normalization constants
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.size = size
        
        # Build transform (resize if needed, then normalize)
        transform_list = []
        if size != 32:
            transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.ToTensor())  # This also does min-max normalization (divides by 255)
        transform_list.append(transforms.Normalize(self.mean, self.std))
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image (H, W, C) format with values in [0, 255]
        img = self.data[idx]
        
        # Convert to PIL Image for transforms
        from PIL import Image
        img = Image.fromarray(img.astype(np.uint8))
        
        # Apply transforms (ToTensor does min-max normalization by dividing by 255)
        img = self.transform(img)
        
        # Return image and dummy label (-1 since we don't have labels for corrupted data)
        return img, -1


def train(epoch, net, optimizer, trainloader, criterion, device):
    """Training function."""
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)


def extract_representations(net, dataloader, criterion, device, compute_acc=True):
    """Extract representations and optionally compute validation metrics."""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    latents = []
    latent_shape = None
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Run model with return_latent=True
            outputs, latent = net(inputs, return_latent=True)
            
            # Capture shape from first batch
            if batch_idx == 0:
                latent_shape = list(latent.shape[1:])
            
            # Compute loss and accuracy only if we have valid labels
            if compute_acc and targets[0].item() != -1:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            
            # Collect latent representations
            latents.append(latent.cpu().numpy())
    
    # Concatenate all latent representations
    representations = np.concatenate(latents, axis=0)
    
    if compute_acc and total > 0:
        acc = 100. * correct / total
        avg_loss = test_loss / (batch_idx + 1)
    else:
        acc = None
        avg_loss = None
    
    return representations, avg_loss, acc, latent_shape[0] if latent_shape else None


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup experiment name and directories
    if args.custom_initialization:
        experiment_name = f"{args.net}_{args.dataset}_custom_init_gain{args.init_gain}_lr{args.lr}_epochs{args.n_epochs}_seed{args.seed}"
    else:
        experiment_name = f"{args.net}_{args.dataset}_lr{args.lr}_epochs{args.n_epochs}_seed{args.seed}"
    
    results_dir = os.path.join("experiments", "results", "train_model_and_extract_representations", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup wandb
    usewandb = ~args.nowandb
    if usewandb:
        import wandb
        wandb.init(project="ordinal-similarity-metrics",
                name=f"repr_extraction_{experiment_name}")
        wandb.config.update(args)
    
    bs = int(args.bs)
    size = int(args.size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0 

    # Set up normalization based on the dataset
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Dataset must be either 'cifar10' or 'cifar100'")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Prepare dataset
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)

    # Prepare corrupted data loader if path is provided
    corrupted_loader = None
    if args.corrupted_data_path:
        corrupted_dataset = CorruptedCIFAR10Dataset(args.corrupted_data_path, size=size)
        corrupted_loader = torch.utils.data.DataLoader(
            corrupted_dataset, batch_size=bs, shuffle=False, num_workers=8
        )
        print(f"Corrupted dataset loaded with {len(corrupted_dataset)} samples")

    # Model factory based on net and dataset
    print('==> Building model..')
    model_registry = {
        ('vit', 'cifar10'): ViT_CIFAR10,
        ('vit', 'cifar100'): ViT_CIFAR100,
        ('res50', 'cifar10'): ResNet50_CIFAR10,
        ('res50', 'cifar100'): ResNet50_CIFAR100,
    }
    
    model_key = (args.net, args.dataset)
    if model_key not in model_registry:
        raise ValueError(f"Unsupported combination: net={args.net}, dataset={args.dataset}")
    
    model_fn = model_registry[model_key]
    seed_everything(args.seed)
    net = model_fn().to(device)

    if args.custom_initialization:
        seed_everything(args.seed)
        initialize_weights(net, gain=args.init_gain)

    # Loss is CE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)  
        
    # Exponential decay scheduler
    gamma = (args.min_lr / args.lr) ** (1.0 / args.n_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Storage for representations and training logs
    epoch_representations = {}  # epoch -> representations
    epoch_corrupted_representations = {}  # epoch -> corrupted representations
    training_logs = []
    
    # Epoch -1: Extract representations before training
    print('\nEpoch: -1 (before training)')
    start = time.time()
    representations, val_loss, acc, latent_shape = extract_representations(
        net, testloader, criterion, device
    )
    epoch_representations[-1] = representations
    
    # Compute pairwise euclidean distance stats for test set
    real_dist_stats = compute_pairwise_euclidean_distance_stats(representations)
    
    # Extract corrupted representations if available
    corrupted_dist_stats = None
    combined_dist_stats = None
    if corrupted_loader:
        corrupted_repr, _, _, _ = extract_representations(
            net, corrupted_loader, criterion, device, compute_acc=False
        )
        epoch_corrupted_representations[-1] = corrupted_repr
        corrupted_dist_stats = compute_pairwise_euclidean_distance_stats(corrupted_repr)
        
        # Compute pairwise distance stats for combined real + corrupted
        combined_repr = np.concatenate([representations, corrupted_repr], axis=0)
        combined_dist_stats = compute_pairwise_euclidean_distance_stats(combined_repr)
    
    log_dict = {
        "epoch": -1, 
        "val_loss": val_loss, 
        "val_acc": acc, 
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time() - start,
        "latent_shape": latent_shape,
        "real_pairwise_euclidean_dist_min": real_dist_stats['min'],
        "real_pairwise_euclidean_dist_max": real_dist_stats['max'],
        "real_pairwise_euclidean_dist_avg": real_dist_stats['avg'],
    }
    if corrupted_dist_stats is not None:
        log_dict["corrupted_pairwise_euclidean_dist_min"] = corrupted_dist_stats['min']
        log_dict["corrupted_pairwise_euclidean_dist_max"] = corrupted_dist_stats['max']
        log_dict["corrupted_pairwise_euclidean_dist_avg"] = corrupted_dist_stats['avg']
    if combined_dist_stats is not None:
        log_dict["real_and_corrupted_pairwise_euclidean_dist_min"] = combined_dist_stats['min']
        log_dict["real_and_corrupted_pairwise_euclidean_dist_max"] = combined_dist_stats['max']
        log_dict["real_and_corrupted_pairwise_euclidean_dist_avg"] = combined_dist_stats['avg']
    training_logs.append(log_dict)
    print(f"Epoch -1: Val Loss: {val_loss:.4f}, Val Acc: {acc:.2f}%")
    
    if usewandb:
        wandb.log(log_dict)

    # Training loop
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        
        # Train the model
        trainloss = train(epoch, net, optimizer, trainloader, criterion, device)
        
        # Extract representations from test set
        representations, val_loss, acc, latent_shape = extract_representations(
            net, testloader, criterion, device
        )
        epoch_representations[epoch] = representations
        
        # Compute pairwise euclidean distance stats for test set
        real_dist_stats = compute_pairwise_euclidean_distance_stats(representations)
        
        # Extract corrupted representations if available
        corrupted_dist_stats = None
        combined_dist_stats = None
        if corrupted_loader:
            corrupted_repr, _, _, _ = extract_representations(
                net, corrupted_loader, criterion, device, compute_acc=False
            )
            epoch_corrupted_representations[epoch] = corrupted_repr
            corrupted_dist_stats = compute_pairwise_euclidean_distance_stats(corrupted_repr)
            
            # Compute pairwise distance stats for combined real + corrupted
            combined_repr = np.concatenate([representations, corrupted_repr], axis=0)
            combined_dist_stats = compute_pairwise_euclidean_distance_stats(combined_repr)
        
        scheduler.step()
        
        # Log training
        log_dict = {
            "epoch": epoch, 
            "train_loss": trainloss, 
            "val_loss": val_loss, 
            "val_acc": acc, 
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time() - start,
            "latent_shape": latent_shape,
            "real_pairwise_euclidean_dist_min": real_dist_stats['min'],
            "real_pairwise_euclidean_dist_max": real_dist_stats['max'],
            "real_pairwise_euclidean_dist_avg": real_dist_stats['avg'],
        }
        if corrupted_dist_stats is not None:
            log_dict["corrupted_pairwise_euclidean_dist_min"] = corrupted_dist_stats['min']
            log_dict["corrupted_pairwise_euclidean_dist_max"] = corrupted_dist_stats['max']
            log_dict["corrupted_pairwise_euclidean_dist_avg"] = corrupted_dist_stats['avg']
        if combined_dist_stats is not None:
            log_dict["real_and_corrupted_pairwise_euclidean_dist_min"] = combined_dist_stats['min']
            log_dict["real_and_corrupted_pairwise_euclidean_dist_max"] = combined_dist_stats['max']
            log_dict["real_and_corrupted_pairwise_euclidean_dist_avg"] = combined_dist_stats['avg']
        training_logs.append(log_dict)
        print(f"Epoch {epoch}: Train Loss: {trainloss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.2f}%, Real AvgDist: {real_dist_stats['avg']:.4f}")
        
        if usewandb:
            wandb.log(log_dict)

    # Save training logs
    training_df = pd.DataFrame(training_logs)
    training_csv_path = os.path.join(results_dir, "training_logs.csv")
    training_df.to_csv(training_csv_path, index=False)
    print(f"\nTraining logs saved to: {training_csv_path}")
    
    # Save representations
    if args.save_representations:
        # Save test set representations
        repr_path = os.path.join(results_dir, "representations.npz")
        np.savez_compressed(repr_path, **{f"epoch_{e}": r for e, r in epoch_representations.items()})
        print(f"Test set representations saved to: {repr_path}")
        
        # Save corrupted representations if available
        if epoch_corrupted_representations:
            corrupted_repr_path = os.path.join(results_dir, "corrupted_representations.npz")
            np.savez_compressed(corrupted_repr_path, **{f"epoch_{e}": r for e, r in epoch_corrupted_representations.items()})
            print(f"Corrupted representations saved to: {corrupted_repr_path}")

    # Save wandb
    if usewandb:
        wandb.save(f"wandb_{experiment_name}.h5")
        wandb.finish()

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
