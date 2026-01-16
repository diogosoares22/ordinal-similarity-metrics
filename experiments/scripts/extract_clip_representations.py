#!/usr/bin/env python3
"""
Script to extract CLIP representations from the COCO Captions dataset.

For each CLIP model (small, medium, large), extracts:
- Image representations using model.encode()
- Caption representations using model.encode()

Saves representations to separate files for images and captions.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from torchvision.datasets import CocoCaptions
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
        description='Extract CLIP representations from COCO Captions dataset'
    )
    parser.add_argument(
        '--coco-root', 
        type=str, 
        default='./data/coco',
        help='Path to COCO dataset root directory'
    )
    parser.add_argument(
        '--coco-ann-file',
        type=str,
        default='./data/coco/annotations/captions_val2017.json',
        help='Path to COCO captions annotation file'
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
        '--caption-index',
        type=int,
        default=0,
        help='Which caption to use per image (COCO has 5 captions per image)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified.'
    )
    return parser.parse_args()


def load_coco_dataset(coco_root: str, ann_file: str):
    """Load the COCO Captions dataset."""
    print(f"Loading COCO Captions dataset from: {coco_root}")
    print(f"Annotations file: {ann_file}")
    
    # CocoCaptions returns (image, captions) where captions is a list of strings
    dataset = CocoCaptions(
        root=coco_root,
        annFile=ann_file,
    )
    
    print(f"Dataset loaded with {len(dataset)} images")
    return dataset


def extract_representations(
    model: SentenceTransformer,
    dataset: CocoCaptions,
    batch_size: int,
    max_samples: int | None,
    caption_index: int,
    device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract image and caption representations using CLIP model.
    
    Args:
        model: SentenceTransformer CLIP model
        dataset: COCO Captions dataset
        batch_size: Batch size for encoding
        max_samples: Maximum number of samples to process
        caption_index: Which caption to use (0-4)
        device: Device to use for encoding
    
    Returns:
        Tuple of (image_representations, caption_representations)
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Extracting representations for {n_samples} samples...")
    print(f"Using caption index: {caption_index}")
    
    # Collect images and captions
    images = []
    captions = []
    
    print("Loading images and captions...")
    for idx in tqdm(range(n_samples), desc="Loading data"):
        image, caption_list = dataset[idx]
        images.append(image)
        # Use specified caption index, fallback to first if not available
        cap_idx = min(caption_index, len(caption_list) - 1)
        captions.append(caption_list[cap_idx])
    
    # Encode images in batches
    print("\nEncoding images...")
    image_representations = model.encode(
        images,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    # Encode captions in batches
    print("\nEncoding captions...")
    caption_representations = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    return image_representations, caption_representations


def save_representations(
    image_repr: np.ndarray,
    caption_repr: np.ndarray,
    output_dir: Path,
    model_name: str,
    model_size: str
):
    """Save representations to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image representations
    image_path = output_dir / f"coco_image_representations_{model_size}.npy"
    np.save(image_path, image_repr)
    print(f"Saved image representations to: {image_path}")
    print(f"  Shape: {image_repr.shape}")
    
    # Save caption representations
    caption_path = output_dir / f"coco_caption_representations_{model_size}.npy"
    np.save(caption_path, caption_repr)
    print(f"Saved caption representations to: {caption_path}")
    print(f"  Shape: {caption_repr.shape}")
    
    # Save metadata
    metadata_path = output_dir / f"coco_representations_metadata_{model_size}.npz"
    np.savez(
        metadata_path,
        model_name=model_name,
        model_size=model_size,
        n_samples=image_repr.shape[0],
        image_dim=image_repr.shape[1],
        caption_dim=caption_repr.shape[1]
    )
    print(f"Saved metadata to: {metadata_path}")


def main():
    """Main function to extract CLIP representations from COCO dataset."""
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
    
    # Load COCO dataset once
    dataset = load_coco_dataset(args.coco_root, args.coco_ann_file)
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
        
        # Extract representations
        image_repr, caption_repr = extract_representations(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            caption_index=args.caption_index,
            device=device
        )
        
        # Save representations
        save_representations(
            image_repr=image_repr,
            caption_repr=caption_repr,
            output_dir=output_dir,
            model_name=model_name,
            model_size=model_size
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
        print(f"    - coco_image_representations_{model_size}.npy")
        print(f"    - coco_caption_representations_{model_size}.npy")
        print(f"    - coco_representations_metadata_{model_size}.npz")


if __name__ == "__main__":
    main()
