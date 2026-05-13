"""
dataset.py — Dataset Loading and Preprocessing
===============================================
Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
Authors: Munaza Tariq, Amaim Anwar, Areeba Arshad | FAST-NUCES
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, datasets
from transformers import ViTImageProcessor
import numpy as np
import medmnist
from medmnist import BloodMNIST


# ─────────────────────────────────────────────
# ImageNet normalization via HuggingFace processor
# ─────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
MEAN = processor.image_mean   # [0.485, 0.456, 0.406]
STD  = processor.image_std    # [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Transform Pipelines
# ─────────────────────────────────────────────
def get_train_transform(rotation=10, brightness=0.2, contrast=0.2, saturation=0.0):
    """
    Training transforms with augmentation.
    Applied only to training images.
    """
    jitter_params = {"brightness": brightness, "contrast": contrast}
    if saturation > 0:
        jitter_params["saturation"] = saturation

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(**jitter_params),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_eval_transform():
    """
    Evaluation transforms — NO augmentation.
    Applied to validation and test images.
    Ensures reproducible, unbiased evaluation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ─────────────────────────────────────────────
# Kermany Dataset
# ─────────────────────────────────────────────
def get_kermany_datasets(data_dir, val_split=0.2, seed=42):
    """
    Load Kermany Chest X-Ray dataset with proper train/val split.

    PHASE 3 FIX: Loads training directory TWICE with different transforms.
    This prevents the Phase 2 bug where validation images were augmented.

    Args:
        data_dir (str): Root directory containing train/ and test/ folders
        val_split (float): Fraction of training data for validation (default 0.2)
        seed (int): Random seed for reproducibility (default 42)

    Returns:
        train_ds, val_ds, test_ds: PyTorch Dataset objects
        class_names: List of class names
    """
    train_dir = f"{data_dir}/train"
    test_dir  = f"{data_dir}/test"

    train_transform = get_train_transform(rotation=10, brightness=0.2, contrast=0.2)
    eval_transform  = get_eval_transform()

    # Load training directory TWICE — different transforms
    # This is the Phase 3 validation bug fix
    full_train_aug  = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_transform)

    # Load test set
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = full_train_aug.classes  # ['NORMAL', 'PNEUMONIA']

    # Create reproducible 80/20 split
    total_size = len(full_train_aug)
    val_size   = int(val_split * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(total_size, generator=generator)

    train_indices = indices[:train_size].tolist()
    val_indices   = indices[train_size:].tolist()

    # Training subset → gets augmentation
    train_ds = Subset(full_train_aug, train_indices)
    # Validation subset → gets eval transform (NO augmentation) ← Phase 3 fix
    val_ds   = Subset(full_train_eval, val_indices)

    print(f"Dataset: Kermany Chest X-Ray")
    print(f"  Classes:    {class_names}")
    print(f"  Train:      {len(train_ds)} images")
    print(f"  Validation: {len(val_ds)} images")
    print(f"  Test:       {len(test_ds)} images")

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(dataset, class_names):
    """
    Compute inverse-frequency class weights to handle class imbalance.

    Formula: w_c = N / (num_classes * N_c)
    where N = total samples, N_c = samples in class c

    Args:
        dataset: Training dataset (Subset of ImageFolder)
        class_names: List of class names

    Returns:
        class_weights (torch.Tensor): Weight per class
        sample_weights (list): Weight per sample (for WeightedRandomSampler)
    """
    # Count samples per class
    targets = [dataset.dataset.targets[i] for i in dataset.indices]
    class_counts = torch.bincount(torch.tensor(targets))
    total = len(targets)
    num_classes = len(class_names)

    # Inverse frequency weights
    class_weights = total / (num_classes * class_counts.float())

    # Per-sample weights for WeightedRandomSampler
    sample_weights = [class_weights[t].item() for t in targets]

    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        print(f"  {name}: {count} samples, weight = {weight:.4f}")

    return class_weights, sample_weights


def get_kermany_loaders(data_dir, batch_size=32, val_split=0.2, seed=42):
    """
    Create DataLoaders for Kermany dataset with class balancing.

    Args:
        data_dir (str): Root data directory
        batch_size (int): Batch size for all loaders
        val_split (float): Validation fraction
        seed (int): Random seed

    Returns:
        train_loader, val_loader, test_loader, class_weights, class_names
    """
    train_ds, val_ds, test_ds, class_names = get_kermany_datasets(
        data_dir, val_split=val_split, seed=seed
    )

    print("\nComputing class weights:")
    class_weights, sample_weights = compute_class_weights(train_ds, class_names)

    # WeightedRandomSampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights, class_names


# ─────────────────────────────────────────────
# BloodMNIST Dataset
# ─────────────────────────────────────────────
def get_bloodmnist_loaders(batch_size=32):
    """
    Create DataLoaders for BloodMNIST (cross-domain evaluation).

    Uses predefined train/val/test splits from MedMNIST.
    No class weighting needed (approximately balanced).

    Args:
        batch_size (int): Batch size for all loaders

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_train_transform(
        rotation=15, brightness=0.2, contrast=0.2, saturation=0.2
    )
    eval_transform = get_eval_transform()

    train_ds = BloodMNIST(split="train", transform=train_transform,
                          download=True, size=224)
    val_ds   = BloodMNIST(split="val",   transform=eval_transform,
                          download=True, size=224)
    test_ds  = BloodMNIST(split="test",  transform=eval_transform,
                          download=True, size=224)

    print(f"Dataset: BloodMNIST")
    print(f"  Classes:    8 blood cell types")
    print(f"  Train:      {len(train_ds)} images")
    print(f"  Validation: {len(val_ds)} images")
    print(f"  Test:       {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
