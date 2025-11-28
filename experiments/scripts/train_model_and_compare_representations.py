#!/usr/bin/env python3
"""
Script to train a single model and compare representations across epochs.

At each epoch, saves the representations from the test set.
At the end, computes similarity between representations from epoch i and the final epoch.

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

from src.baselines import run_approximate_baseline_measures
from src.tsi import ApproxTSI
from src.qsi import ApproxQSI
from src.data import RepresentationPair

from models.initialization import initialize_weights
from models.vit import ViT_CIFAR10, ViT_CIFAR100
from models.resnet import ResNet50_CIFAR10, ResNet50_CIFAR100

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
    parser = argparse.ArgumentParser(description='Train model and compare representations across epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--net', default='vit', choices=['vit', 'res50'], help='network architecture')
    parser.add_argument('--bs', default=512)
    parser.add_argument('--size', default=32)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--similarity-bs', default=1000, type=int, help='batch size for similarity computation')
    parser.add_argument('--no-similarity-bs', default=10, type=int, help='number of batches for similarity computation')
    parser.add_argument('--custom-initialization', action='store_true', help='use custom initialization')
    parser.add_argument('--init-gain', default=1, type=float, help='gain for weight initialization')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate for exponential decay')
    parser.add_argument('--save-representations', action='store_true', default=True, help='save representations to disk (can be large)')
    parser.add_argument('--similarity-only', action='store_true', help='skip training and only compute similarity from saved representations')
    parser.add_argument('--representations-path', type=str, default=None, help='path to saved representations.npz file (used with --similarity-only)')
    return parser.parse_args()


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


def extract_representations(net, testloader, criterion, device):
    """Extract representations and compute validation metrics for a single model."""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    latents = []
    latent_shape = None
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Run model with return_latent=True
            outputs, latent = net(inputs, return_latent=True)
            
            # Capture shape from first batch
            if batch_idx == 0:
                latent_shape = list(latent.shape[1:])
            
            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Collect latent representations
            latents.append(latent.cpu().numpy())
    
    # Concatenate all latent representations
    representations = np.concatenate(latents, axis=0)
    
    acc = 100. * correct / total
    avg_loss = test_loss / (batch_idx + 1)
    
    return representations, avg_loss, acc, latent_shape[0] if latent_shape else None


def compute_similarity_scores(X, Y, batch_size, no_batches):
    """Compute similarity scores between two sets of representations."""
    results = run_approximate_baseline_measures(X, Y, batch_size=batch_size, no_batches=no_batches)
    approx_n_samples = (batch_size ** 2) * no_batches
    d = lambda a, b: np.linalg.norm(a - b)
    representations = RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)
    approx_tsi_sampling = ApproxTSI(n_samples=approx_n_samples, n_threads=8)(representations)
    approx_qsi_sampling = ApproxQSI(n_samples=approx_n_samples, n_threads=8)(representations)
    results["C-TSI"] = approx_tsi_sampling
    results["C-QSI"] = approx_qsi_sampling
    return results    


def load_representations_from_file(repr_path):
    """Load representations from a saved .npz file."""
    print(f"==> Loading representations from: {repr_path}")
    data = np.load(repr_path)
    epoch_representations = {}
    for key in data.files:
        # Keys are in format "epoch_-1", "epoch_0", "epoch_1", etc.
        epoch_num = int(key.replace("epoch_", ""))
        epoch_representations[epoch_num] = data[key]
    print(f"Loaded representations for {len(epoch_representations)} epochs: {sorted(epoch_representations.keys())}")
    return epoch_representations


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup experiment name and directories
    if args.custom_initialization:
        experiment_name = f"{args.net}_{args.dataset}_custom_init_gain{args.init_gain}_lr{args.lr}_epochs{args.n_epochs}_seed{args.seed}"
    else:
        experiment_name = f"{args.net}_{args.dataset}_lr{args.lr}_epochs{args.n_epochs}_seed{args.seed}"
    
    results_dir = os.path.join("experiments", "results", "train_model_and_compare_representations", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup wandb
    usewandb = ~args.nowandb
    if usewandb:
        import wandb
        wandb.init(project="ordinal-similarity-metrics",
                name=f"repr_comparison_{experiment_name}")
        wandb.config.update(args)
    
    # Handle similarity-only mode
    if args.similarity_only:
        # Determine path to representations file
        if args.representations_path:
            repr_path = args.representations_path
        else:
            repr_path = os.path.join(results_dir, "representations.npz")
        
        if not os.path.exists(repr_path):
            raise FileNotFoundError(f"Representations file not found: {repr_path}. "
                                    "Please run training first or provide --representations-path.")
        
        epoch_representations = load_representations_from_file(repr_path)
    else:
        # Full training mode
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
        training_logs = []
        
        # Epoch -1: Extract representations before training
        print('\nEpoch: -1 (before training)')
        start = time.time()
        representations, val_loss, acc, latent_shape = extract_representations(
            net, testloader, criterion, device
        )
        epoch_representations[-1] = representations
        
        log_dict = {
            "epoch": -1, 
            "val_loss": val_loss, 
            "val_acc": acc, 
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time() - start,
            "latent_shape": latent_shape,
        }
        training_logs.append(log_dict)
        print(f"Epoch -1: Val Loss: {val_loss:.4f}, Val Acc: {acc:.2f}%")
        
        if usewandb:
            wandb.log(log_dict)

        # Training loop
        for epoch in range(start_epoch, args.n_epochs):
            start = time.time()
            
            # Train the model
            trainloss = train(epoch, net, optimizer, trainloader, criterion, device)
            
            # Extract representations
            representations, val_loss, acc, latent_shape = extract_representations(
                net, testloader, criterion, device
            )
            epoch_representations[epoch] = representations
            
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
            }
            training_logs.append(log_dict)
            print(f"Epoch {epoch}: Train Loss: {trainloss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.2f}%")
            
            if usewandb:
                wandb.log(log_dict)

        # Save training logs
        training_df = pd.DataFrame(training_logs)
        training_csv_path = os.path.join(results_dir, "training_logs.csv")
        training_df.to_csv(training_csv_path, index=False)
        print(f"\nTraining logs saved to: {training_csv_path}")
        
        # Optionally save representations
        if args.save_representations:
            repr_path = os.path.join(results_dir, "representations.npz")
            np.savez_compressed(repr_path, **{f"epoch_{e}": r for e, r in epoch_representations.items()})
            print(f"Representations saved to: {repr_path}")

    # Compute similarity between each epoch and the final epoch
    print("\n==> Computing similarities between each epoch and final epoch...")
    all_epochs = sorted(epoch_representations.keys())
    final_epoch = max(all_epochs)  # Use the highest epoch number as final
    final_representations = epoch_representations[final_epoch]
    print(f"Using epoch {final_epoch} as the final epoch for comparison")
    
    similarity_logs = []
    
    for epoch in all_epochs:
        if epoch == final_epoch:
            # Similarity with itself (should be perfect)
            similarity_scores = compute_similarity_scores(
                final_representations, final_representations,
                batch_size=args.similarity_bs, no_batches=args.no_similarity_bs
            )
        else:
            similarity_scores = compute_similarity_scores(
                epoch_representations[epoch], final_representations,
                batch_size=args.similarity_bs, no_batches=args.no_similarity_bs
            )
        
        log_entry = {"epoch": epoch, "compared_to": final_epoch}
        for metric_name, metric_value in similarity_scores.items():
            log_entry[metric_name] = metric_value
        similarity_logs.append(log_entry)
        
        if usewandb:
            wandb_log = {"epoch_comparison": epoch}
            for metric_name, metric_value in similarity_scores.items():
                if metric_value is not None:
                    wandb_log[f"similarity_to_final/{metric_name}"] = metric_value
            wandb.log(wandb_log)

    # Save similarity results
    similarity_df = pd.DataFrame(similarity_logs)
    similarity_csv_path = os.path.join(results_dir, "similarity_to_final_epoch.csv")
    similarity_df.to_csv(similarity_csv_path, index=False)
    print(f"\nSimilarity results saved to: {similarity_csv_path}")

    # Save wandb
    if usewandb:
        wandb.save(f"wandb_{experiment_name}.h5")
        wandb.finish()

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()

