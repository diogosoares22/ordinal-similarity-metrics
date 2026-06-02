#!/usr/bin/env python3
"""
Script to extract CLIP representations from the ImageNet dataset.

For each CLIP model (small, medium, large), extracts:
- Image representations using model.encode()
- Text representations using "A photo of {label}" for each class

Saves representations to separate files for images and text.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from torchvision.datasets import ImageNet
from PIL import Image


# CLIP models available via sentence-transformers
CLIP_MODELS = {
    "small": "clip-ViT-B-32",
    "medium": "clip-ViT-B-16",
    "large": "clip-ViT-L-14"
}


def initialize_clip_model(model_name: str) -> SentenceTransformer:
    """Initialize and return a CLIP model from sentence-transformers."""
    return SentenceTransformer(model_name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract CLIP representations from ImageNet dataset'
    )
    parser.add_argument(
        '--imagenet-root', 
        type=str, 
        default='./data/imagenet',
        help='Path to ImageNet dataset root directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Which split to use (train or val)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results/clip_representations',
        help='Directory to save extracted representations'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['small', 'medium', 'large', 'all'],
        default=['all'],
        help='Which CLIP models to use (small, medium, large, or all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None for all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified.'
    )
    return parser.parse_args()


def load_imagenet_dataset(imagenet_root: str, split: str):
    """Load the ImageNet dataset."""
    print(f"Loading ImageNet dataset from: {imagenet_root}")
    print(f"Split: {split}")
    
    dataset = ImageNet(
        root=imagenet_root,
        split=split,
    )
    
    print(f"Dataset loaded with {len(dataset)} images")
    print(f"Number of classes: {len(dataset.classes)}")
    return dataset


def print_all_class_names(dataset: ImageNet):
    """Print all class names and their corresponding text prompts for verification."""
    print("\n" + "=" * 60)
    print("CLASS NAMES AND TEXT PROMPTS FOR VERIFICATION")
    print("=" * 60)
    print(f"{'Index':<8} {'Synset ID':<12} {'Text Prompt'}")
    print("-" * 100)
    
    for class_idx in range(len(dataset.classes)):
        class_info = dataset.classes[class_idx]
        synset_id = class_info[0]
        text_prompt = f"A photo of {synset_id}"
        print(f"{class_idx:<8} {synset_id:<12} {text_prompt}")
    
    print("-" * 100)
    print(f"Total classes: {len(dataset.classes)}")
    print("=" * 60 + "\n")


def get_class_name(dataset: ImageNet, class_idx: int) -> str:
    """
    Get the human-readable class name for a given class index.
    
    ImageNet classes are tuples of (synset_id, class_names) where class_names
    is a tuple of synonyms. We use the first (primary) name.
    """
    # dataset.classes is a list of tuples: (synset_id, (name1, name2, ...))
    # We want the first name from the tuple of names
    class_info = dataset.classes[class_idx]
    class_name = class_info[0]
    return class_name


def load_images_and_prompts(
    dataset: ImageNet,
    max_samples: int | None,
) -> tuple[list[Image.Image], list[str], np.ndarray]:
    """
    Load images and generate text prompts from ImageNet dataset.
    
    This is done once before processing multiple models for efficiency.
    
    Args:
        dataset: ImageNet dataset
        max_samples: Maximum number of samples to process
    
    Returns:
        Tuple of (images, text_prompts, labels)
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Loading {n_samples} images and generating text prompts...")
    
    images = []
    text_prompts = []
    labels = []
    
    for idx in tqdm(range(n_samples), desc="Loading data"):
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
        
        # Generate text prompt: "A photo of {class_name}"
        class_name = get_class_name(dataset, label)
        text_prompt = f"A photo of {class_name}"
        text_prompts.append(text_prompt)
    
    print(f"Loaded {len(images)} images and {len(text_prompts)} text prompts")
    
    return images, text_prompts, np.array(labels)


def encode_representations(
    model: SentenceTransformer,
    images: list[Image.Image],
    text_prompts: list[str],
    batch_size: int,
    device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode images and text prompts using CLIP model.
    
    Args:
        model: SentenceTransformer CLIP model
        images: List of PIL images
        text_prompts: List of text prompts
        batch_size: Batch size for encoding
        device: Device to use for encoding
    
    Returns:
        Tuple of (image_representations, text_representations)
    """
    print(f"Encoding {len(images)} samples...")
    
    # Encode images in batches
    print("\nEncoding images...")
    image_representations = model.encode(
        images,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    # Encode text prompts in batches
    print("\nEncoding text prompts...")
    text_representations = model.encode(
        text_prompts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    return image_representations, text_representations


def save_representations(
    image_repr: np.ndarray,
    text_repr: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    model_name: str,
    model_size: str,
    split: str
):
    """Save representations to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image representations
    image_path = output_dir / f"imagenet_{split}_image_representations_{model_size}.npy"
    np.save(image_path, image_repr)
    print(f"Saved image representations to: {image_path}")
    print(f"  Shape: {image_repr.shape}")
    
    # Save text representations
    text_path = output_dir / f"imagenet_{split}_text_representations_{model_size}.npy"
    np.save(text_path, text_repr)
    print(f"Saved text representations to: {text_path}")
    print(f"  Shape: {text_repr.shape}")
    
    # Save labels
    labels_path = output_dir / f"imagenet_{split}_labels_{model_size}.npy"
    np.save(labels_path, labels)
    print(f"Saved labels to: {labels_path}")
    print(f"  Shape: {labels.shape}")
    
    # Save metadata
    metadata_path = output_dir / f"imagenet_{split}_representations_metadata_{model_size}.npz"
    np.savez(
        metadata_path,
        model_name=model_name,
        model_size=model_size,
        split=split,
        n_samples=image_repr.shape[0],
        image_dim=image_repr.shape[1],
        text_dim=text_repr.shape[1]
    )
    print(f"Saved metadata to: {metadata_path}")


def main():
    """Main function to extract CLIP representations from ImageNet dataset."""
    args = parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Determine which models to use
    if 'all' in args.models:
        model_sizes = list(CLIP_MODELS.keys())
    else:
        model_sizes = args.models
    
    print(f"Models to process: {model_sizes}")
    print()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    
    # Load ImageNet dataset once
    dataset = load_imagenet_dataset(args.imagenet_root, args.split)
    
    # Print all class names for verification
    print_all_class_names(dataset)
    
    # Load images and text prompts once (expensive operation)
    print("\n" + "=" * 60)
    print("LOADING IMAGES AND TEXT PROMPTS (done once for all models)")
    print("=" * 60)
    images, text_prompts, labels = load_images_and_prompts(
        dataset=dataset,
        max_samples=args.max_samples
    )
    print()
    
    # Process each model
    for model_size in model_sizes:
        model_name = CLIP_MODELS[model_size]
        print("=" * 60)
        print(f"Processing model: {model_size} ({model_name})")
        print("=" * 60)
        
        # Initialize model
        print(f"Loading model: {model_name}")
        model = initialize_clip_model(model_name)
        
        # Encode representations
        image_repr, text_repr = encode_representations(
            model=model,
            images=images,
            text_prompts=text_prompts,
            batch_size=args.batch_size,
            device=device
        )
        
        # Save representations
        save_representations(
            image_repr=image_repr,
            text_repr=text_repr,
            labels=labels,
            output_dir=output_dir,
            model_name=model_name,
            model_size=model_size,
            split=args.split
        )
        
        # Clear model from memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print()
    
    print("=" * 60)
    print("All representations extracted successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Summary
    print("\nOutput files:")
    for model_size in model_sizes:
        print(f"  {model_size}:")
        print(f"    - imagenet_{args.split}_image_representations_{model_size}.npy")
        print(f"    - imagenet_{args.split}_text_representations_{model_size}.npy")
        print(f"    - imagenet_{args.split}_labels_{model_size}.npy")
        print(f"    - imagenet_{args.split}_representations_metadata_{model_size}.npz")


if __name__ == "__main__":
    main()
