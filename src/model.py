"""
model.py — ViT Model Definition and Layer Freezing
===================================================
Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
Authors: Munaza Tariq, Amaim Anwar, Areeba Arshad | FAST-NUCES
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


def load_vit_model(num_labels=2, dropout=0.1):
    """
    Load pretrained ViT-Base-Patch16-224 from HuggingFace.
    Replace the 1000-class ImageNet head with num_labels-class head.

    Args:
        num_labels (int): Number of output classes (2 for Kermany, 8 for BloodMNIST)
        dropout (float): Dropout probability for hidden and attention layers

    Returns:
        model: ViTForImageClassification with modified classification head
    """
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,      # allows head replacement
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
    )
    return model


def apply_phase2_config(model):
    """
    Phase 2 Configuration — Full Fine-Tuning (Baseline Reproduction).
    All 86 million parameters are trainable.
    No layer freezing applied.

    Args:
        model: ViTForImageClassification instance

    Returns:
        model: All parameters trainable
    """
    # Ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Phase 2] Total parameters:     {total:,}")
    print(f"[Phase 2] Trainable parameters: {trainable:,}")
    print(f"[Phase 2] Frozen parameters:    {total - trainable:,}")
    return model


def apply_phase3_config(model, freeze_blocks=list(range(10))):
    """
    Phase 3 Configuration — Partial Fine-Tuning (Proposed Method).

    Freezes:
      - Patch embedding layer (vit.embeddings)
      - Encoder blocks 0-9 (first 10 of 12)

    Keeps trainable:
      - Encoder blocks 10 and 11
      - Final LayerNorm (vit.layernorm)
      - Classification head (classifier)

    Result: ~86M → ~14M trainable parameters (84% reduction)

    Args:
        model: ViTForImageClassification instance
        freeze_blocks (list): List of encoder block indices to freeze

    Returns:
        model: With selective parameter freezing applied
    """
    # Step 1 — Freeze patch embedding layer
    # (patch projection, positional embeddings, CLS token)
    for param in model.vit.embeddings.parameters():
        param.requires_grad = False

    # Step 2 — Freeze specified encoder blocks (default: 0-9)
    for block_idx in freeze_blocks:
        for param in model.vit.encoder.layer[block_idx].parameters():
            param.requires_grad = False

    # Step 3 — Verify trainable components
    # (blocks 10, 11, layernorm, classifier are already requires_grad=True)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"[Phase 3] Total parameters:     {total:,}")
    print(f"[Phase 3] Trainable parameters: {trainable:,}")
    print(f"[Phase 3] Frozen parameters:    {frozen:,}")
    print(f"[Phase 3] Parameter reduction:  {frozen/total*100:.1f}%")
    print(f"[Phase 3] Frozen blocks:        {freeze_blocks}")
    print(f"[Phase 3] Trainable blocks:     [10, 11]")

    return model


def get_model(phase, num_labels=2, dropout=0.1):
    """
    Factory function — returns configured model for given phase.

    Args:
        phase (int): 2 for baseline, 3 for proposed method
        num_labels (int): Number of output classes
        dropout (float): Dropout probability

    Returns:
        model: Configured ViTForImageClassification
    """
    model = load_vit_model(num_labels=num_labels, dropout=dropout)

    if phase == 2:
        model = apply_phase2_config(model)
    elif phase == 3:
        model = apply_phase3_config(model, freeze_blocks=list(range(10)))
    else:
        raise ValueError(f"Phase must be 2 or 3. Got: {phase}")

    return model


def count_parameters(model):
    """
    Utility — count and display model parameters.

    Args:
        model: PyTorch model

    Returns:
        dict: total, trainable, frozen parameter counts
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    frozen,
        "reduction": f"{frozen/total*100:.1f}%"
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Phase 2 Model Configuration")
    print("=" * 50)
    model_p2 = get_model(phase=2, num_labels=2)
    params_p2 = count_parameters(model_p2)
    print(f"Summary: {params_p2}")

    print("\n" + "=" * 50)
    print("Testing Phase 3 Model Configuration")
    print("=" * 50)
    model_p3 = get_model(phase=3, num_labels=2)
    params_p3 = count_parameters(model_p3)
    print(f"Summary: {params_p3}")
