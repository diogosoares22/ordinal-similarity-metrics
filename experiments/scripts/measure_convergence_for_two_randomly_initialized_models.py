#!/usr/bin/env python3
"""
Script to measure the convergence of two randomly initialized models.

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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--net', default='vit', choices=['vit', 'res50'], help='network architecture')
    parser.add_argument('--bs', default=512)
    parser.add_argument('--size', default=32)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--similarity-bs', default=1000, type=int, help='batch size for similarity computation')
    parser.add_argument('--no-similarity-bs', default=10, type=int, help='number of batches for similarity computation')
    parser.add_argument('--init-gain', default=10.0, type=float, help='gain for weight initialization')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimum learning rate for exponential decay')
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


def test(epoch, net0, net1, testloader, criterion, device, batch_size=256, no_batches=10):
    """Validation function that also computes similarity between representations."""
    net0.eval()
    net1.eval()
    test_loss0 = 0
    test_loss1 = 0
    correct0 = 0
    correct1 = 0
    total = 0
    
    latents0 = []
    latents1 = []
    
    # Store shapes of datapoints (captured from first batch)
    latent0_shape = None
    latent1_shape = None
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Run both models with return_latent=True
            outputs0, latent0 = net0(inputs, return_latent=True)
            outputs1, latent1 = net1(inputs, return_latent=True)
            
            # Capture shapes from first batch (single datapoint shape, excluding batch dim)
            if batch_idx == 0:
                latent0_shape = list(latent0.shape[1:])  # e.g., [512]
                latent1_shape = list(latent1.shape[1:])  # e.g., [512]
            
            # Compute losses
            loss0 = criterion(outputs0, targets)
            loss1 = criterion(outputs1, targets)

            test_loss0 += loss0.item()
            test_loss1 += loss1.item()
            
            # Compute accuracy for net0
            _, predicted0 = outputs0.max(1)
            correct0 += predicted0.eq(targets).sum().item()
            
            # Compute accuracy for net1
            _, predicted1 = outputs1.max(1)
            correct1 += predicted1.eq(targets).sum().item()
            
            total += targets.size(0)
            
            # Collect latent representations
            latents0.append(latent0.cpu().numpy())
            latents1.append(latent1.cpu().numpy())
    
    # Concatenate all latent representations
    X = np.concatenate(latents0, axis=0)
    Y = np.concatenate(latents1, axis=0)

    # Compute similarity scores
    similarity_scores = obtain_similarity_scores(X, Y, batch_size=batch_size, no_batches=no_batches)
    
    acc0 = 100. * correct0 / total
    acc1 = 100. * correct1 / total
    
    # Return shapes info as well
    shapes_info = {
        "latent0_shape": latent0_shape[0],
        "latent1_shape": latent1_shape[0],
    }

    return test_loss0, acc0, test_loss1, acc1, similarity_scores, shapes_info

def obtain_similarity_scores(X, Y, batch_size, no_batches):
    results = run_approximate_baseline_measures(X, Y, batch_size=batch_size, no_batches=no_batches)
    approx_n_samples = (batch_size ** 2) * no_batches
    d = lambda a, b: np.linalg.norm(a - b)
    representations = RepresentationPair(X=X, Y=Y, d_x=d, d_y=d)
    approx_tsi_sampling = ApproxTSI(n_samples=approx_n_samples, n_threads=8)(representations)
    approx_qsi_sampling = ApproxQSI(n_samples=approx_n_samples, n_threads=8)(representations)
    results["C-TSI"] = approx_tsi_sampling
    results["C-QSI"] = approx_qsi_sampling
    return results    

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup wandb
    usewandb = ~args.nowandb
    if usewandb:
        import wandb
        watermark = "{}_{}_gain{}_lr{}_epochs{}".format(args.net, args.dataset, args.init_gain, args.lr, args.n_epochs)
        wandb.init(project="ordinal-similarity-metrics",
                name=watermark)
        wandb.config.update(args)
    
    # Setup CSV logging
    results_dir = os.path.join("experiments", "results", "measure_convergence_for_two_randomly_initialized_models")
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = f"{args.net}_{args.dataset}_seed{args.seed}_epochs{args.n_epochs}_gain{args.init_gain}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    all_logs = []  # Collect all log dictionaries

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

    # Set up class names based on the dataset
    if args.dataset == 'cifar10':
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        # CIFAR100 has 100 classes, so we don't list them all here
        classes = None

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
    net0 = model_fn().to(device)
    net1 = model_fn().to(device)

    seed_everything(args.seed)
    initialize_weights(net0, gain=args.init_gain)
    seed_everything(args.seed + 1)
    initialize_weights(net1, gain=args.init_gain)

    # Loss is CE
    criterion = nn.CrossEntropyLoss()
    optimizer0 = optim.Adam(net0.parameters(), lr=args.lr)  
    optimizer1 = optim.Adam(net1.parameters(), lr=args.lr)  
        
    # Exponential decay scheduler: lr * gamma^epoch, where gamma is computed to reach min_lr at the end
    # lr * gamma^n_epochs = min_lr  =>  gamma = (min_lr / lr)^(1/n_epochs)
    gamma = (args.min_lr / args.lr) ** (1.0 / args.n_epochs)
    scheduler0 = torch.optim.lr_scheduler.ExponentialLR(optimizer0, gamma=gamma)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=gamma)

    # Epoch -1: Test before training (baseline measurement)
    print('\nEpoch: -1 (before training)')
    start = time.time()
    val_loss0, acc0, val_loss1, acc1, similarity_scores, shapes_info = test(
        -1, net0, net1, testloader, criterion, device,
        batch_size=args.similarity_bs, no_batches=args.no_similarity_bs
    )
    log_dict = {
        "epoch": -1, 
        "val_loss_0": val_loss0, 
        "val_acc_0": acc0, 
        "val_loss_1": val_loss1, 
        "val_acc_1": acc1, 
        "lr": optimizer0.param_groups[0]["lr"],
        "epoch_time": time.time()-start
    }
    for metric_name, metric_value in similarity_scores.items():
        log_dict[f"similarity/{metric_name}"] = metric_value
    # Log shapes of datapoints
    log_dict["shapes/latent_0"] = shapes_info["latent0_shape"]
    log_dict["shapes/latent_1"] = shapes_info["latent1_shape"]
    all_logs.append(log_dict)
    if usewandb:
        wandb.log(log_dict)

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        
        # Train both models
        trainloss0 = train(epoch, net0, optimizer0, trainloader, criterion, device)
        trainloss1 = train(epoch, net1, optimizer1, trainloader, criterion, device)
        
        # Test both models and compute similarity scores
        val_loss0, acc0, val_loss1, acc1, similarity_scores, shapes_info = test(
            epoch, net0, net1, testloader, criterion, device,
            batch_size=args.similarity_bs, no_batches=args.no_similarity_bs
        )
        
        scheduler0.step()  # step exponential scheduling
        scheduler1.step()  # step exponential scheduling
        
        # Log training..
        log_dict = {
            "epoch": epoch, 
            "train_loss_0": trainloss0, 
            "val_loss_0": val_loss0, 
            "val_acc_0": acc0, 
            "train_loss_1": trainloss1, 
            "val_loss_1": val_loss1, 
            "val_acc_1": acc1, 
            "lr": optimizer0.param_groups[0]["lr"],
            "epoch_time": time.time()-start
        }
        # Add similarity scores to log
        for metric_name, metric_value in similarity_scores.items():
            log_dict[f"similarity/{metric_name}"] = metric_value
        # Log shapes of datapoints
        log_dict["shapes/latent_0"] = shapes_info["latent0_shape"]
        log_dict["shapes/latent_1"] = shapes_info["latent1_shape"]
        all_logs.append(log_dict)
        if usewandb:
            wandb.log(log_dict)

    # Save logs to CSV
    df = pd.DataFrame(all_logs)
    df.to_csv(csv_path, index=False)
    print(f"\nLogs saved to: {csv_path}")

    # writeout wandb
    if usewandb:
        wandb.save("wandb_{}_{}.h5".format(args.net, args.dataset))


if __name__ == "__main__":
    main()