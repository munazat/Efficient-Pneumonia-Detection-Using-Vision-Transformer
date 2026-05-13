"""
train.py — Main Training Script
================================
Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
Authors: Munaza Tariq, Amaim Anwar, Areeba Arshad | FAST-NUCES

Usage:
    python train.py --phase 2                     # Train Phase 2 baseline
    python train.py --phase 3                     # Train Phase 3 proposed method
    python train.py --phase 3 --dataset bloodmnist  # Train on BloodMNIST
"""

import argparse
import torch
import torch.nn as nn
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model   import get_model
from src.dataset import get_kermany_loaders, get_bloodmnist_loaders
from src.utils   import (
    EarlyStopping, train_one_epoch, evaluate,
    compute_binary_metrics, compute_multiclass_metrics,
    bootstrap_confidence_intervals,
    save_metrics, save_training_log, set_seed
)


def main(args):
    # ── Load config ──────────────────────────────────
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # ── Setup ────────────────────────────────────────
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    phase_cfg = cfg[f"phase{args.phase}"]
    is_binary = (args.dataset == "kermany")

    # ── Load Dataset ─────────────────────────────────
    if args.dataset == "kermany":
        print("\n📂 Loading Kermany Dataset...")
        train_loader, val_loader, test_loader, class_weights, class_names = \
            get_kermany_loaders(
                data_dir   = cfg["data"]["kermany"]["data_dir"],
                batch_size = phase_cfg["training"]["batch_size"],
                val_split  = cfg["data"]["kermany"]["val_split"],
                seed       = cfg["data"]["kermany"]["seed"]
            )
        num_labels = 2

    elif args.dataset == "bloodmnist":
        print("\n📂 Loading BloodMNIST Dataset...")
        train_loader, val_loader, test_loader = get_bloodmnist_loaders(
            batch_size=phase_cfg["training"]["batch_size"]
        )
        class_weights = None
        num_labels    = 8
        is_binary     = False

    # ── Load Model ───────────────────────────────────
    print(f"\n🤖 Loading Model (Phase {args.phase})...")
    model = get_model(phase=args.phase, num_labels=num_labels)
    model = model.to(device)

    # ── Loss Function ────────────────────────────────
    label_smoothing = phase_cfg["loss"]["label_smoothing"]

    if class_weights is not None and phase_cfg["loss"]["use_class_weights"]:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing
        )
        print(f"Loss: CrossEntropyLoss (class weights + label smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"Loss: CrossEntropyLoss (label smoothing={label_smoothing})")

    # ── Optimizer ────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = phase_cfg["optimizer"]["lr"],
        weight_decay = phase_cfg["optimizer"]["weight_decay"]
    )

    # ── LR Scheduler ─────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "min",
        factor  = phase_cfg["scheduler"]["factor"],
        patience= phase_cfg["scheduler"]["patience"]
    )

    # ── Early Stopping ───────────────────────────────
    checkpoint_path = f"checkpoints/phase{args.phase}_{args.dataset}_best.pth"
    early_stopper   = EarlyStopping(
        patience         = phase_cfg["training"]["early_stopping_patience"],
        metric           = "val_f1",
        mode             = "max",
        checkpoint_path  = checkpoint_path
    )

    # ── Training Loop ────────────────────────────────
    max_epochs = phase_cfg["training"]["max_epochs"]
    training_log = []

    print(f"\n🚀 Starting Training — Phase {args.phase} | {args.dataset.upper()}")
    print(f"   Max Epochs:      {max_epochs}")
    print(f"   Early Stopping:  Patience={phase_cfg['training']['early_stopping_patience']}")
    print("=" * 65)

    for epoch in range(1, max_epochs + 1):

        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_results = evaluate(model, val_loader, criterion, device, binary=is_binary)

        # Scheduler step
        scheduler.step(val_results["loss"])

        # Log epoch
        log_entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_results["loss"], 4),
            "val_acc":    round(val_results["acc"], 4),
            "val_f1":     round(val_results["f1"], 4),
        }
        training_log.append(log_entry)

        print(f"Epoch {epoch:2d}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_results['loss']:.4f} | "
              f"Val Acc: {val_results['acc']:.4f} | "
              f"Val F1: {val_results['f1']:.4f}")

        # Early stopping check
        early_stopper.step(val_results["f1"], model)
        if early_stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── Final Evaluation on Test Set ─────────────────
    print(f"\n📊 Final Evaluation on Test Set...")
    print(f"Loading best checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_results = evaluate(model, test_loader, criterion, device, binary=is_binary)

    if is_binary:
        metrics = compute_binary_metrics(
            test_results["labels"],
            test_results["preds"],
            test_results["probs"]
        )
        print("\n" + "=" * 50)
        print(f"TEST RESULTS — Phase {args.phase} | {args.dataset.upper()}")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")

        # Bootstrap CIs
        print("\n📈 Bootstrap Confidence Intervals (1000 resamples):")
        ci_results = bootstrap_confidence_intervals(
            test_results["labels"],
            test_results["preds"],
            test_results["probs"]
        )
        for k, v in ci_results.items():
            print(f"  {k:15s}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")

        metrics["bootstrap_ci"] = ci_results

    else:
        metrics = compute_multiclass_metrics(
            test_results["labels"],
            test_results["preds"],
            test_results["probs"]
        )
        print("\n" + "=" * 50)
        print(f"TEST RESULTS — Phase {args.phase} | {args.dataset.upper()}")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")

    # ── Save Results ─────────────────────────────────
    metrics_path = f"results/phase{args.phase}_{args.dataset}_metrics.json"
    log_path     = f"results/phase{args.phase}_{args.dataset}_training_log.csv"

    save_metrics(metrics, metrics_path)
    save_training_log(training_log, log_path)

    print(f"\n✅ Training complete!")
    print(f"   Metrics:      {metrics_path}")
    print(f"   Training log: {log_path}")
    print(f"   Checkpoint:   {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ViT for Pneumonia Detection"
    )
    parser.add_argument(
        "--phase", type=int, choices=[2, 3], required=True,
        help="Training phase: 2=baseline, 3=proposed method"
    )
    parser.add_argument(
        "--dataset", type=str, default="kermany",
        choices=["kermany", "bloodmnist"],
        help="Dataset to train on (default: kermany)"
    )
    args = parser.parse_args()
    main(args)
