"""
utils.py — Metrics, Early Stopping, and Training Utilities
===========================================================
Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
Authors: Munaza Tariq, Amaim Anwar, Areeba Arshad | FAST-NUCES
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    Saves the best model checkpoint automatically.

    Args:
        patience (int): Number of epochs to wait before stopping
        metric (str): Metric to monitor ('val_f1' or 'val_loss')
        mode (str): 'max' for F1/accuracy, 'min' for loss
        checkpoint_path (str): Path to save best model weights
    """

    def __init__(self, patience=5, metric="val_f1", mode="max",
                 checkpoint_path="checkpoints/best_model.pth"):
        self.patience         = patience
        self.metric           = metric
        self.mode             = mode
        self.checkpoint_path  = checkpoint_path
        self.best_value       = -float("inf") if mode == "max" else float("inf")
        self.epochs_no_improve = 0
        self.should_stop      = False

    def step(self, current_value, model):
        """
        Check if current metric improved. Save model if yes.

        Args:
            current_value (float): Current epoch's metric value
            model: PyTorch model to save if improved

        Returns:
            bool: True if best model was saved this epoch
        """
        improved = (
            current_value > self.best_value if self.mode == "max"
            else current_value < self.best_value
        )

        if improved:
            self.best_value        = current_value
            self.epochs_no_improve = 0
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  ✅ Best model saved! ({self.metric} = {current_value:.4f})")
            return True
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
                print(f"  ⏹️  Early stopping triggered "
                      f"(no improvement for {self.patience} epochs)")
            return False


# ─────────────────────────────────────────────
# Metrics Computation
# ─────────────────────────────────────────────
def compute_binary_metrics(y_true, y_pred, y_probs):
    """
    Compute all metrics for binary classification (Kermany).

    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        y_probs (list): Predicted probabilities for positive class

    Returns:
        dict: accuracy, precision, recall, specificity, f1, auc
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # Specificity = recall of NORMAL class (pos_label=0)
    _, specificity, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=0, zero_division=0
    )
    auc = roc_auc_score(y_true, y_probs)

    return {
        "accuracy":    round(float(accuracy), 4),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score":    round(float(f1), 4),
        "auc_roc":     round(float(auc), 4),
    }


def compute_multiclass_metrics(y_true, y_pred, y_probs, num_classes=8):
    """
    Compute macro-averaged metrics for multi-class classification (BloodMNIST).

    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        y_probs (ndarray): Predicted probabilities for all classes
        num_classes (int): Number of classes

    Returns:
        dict: accuracy, precision, recall, f1, auc (all macro-averaged)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1_score":  round(float(f1), 4),
        "auc_roc":   round(float(auc), 4),
    }


def bootstrap_confidence_intervals(y_true, y_pred, y_probs,
                                   n_resamples=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for binary classification metrics.

    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        y_probs (list): Predicted probabilities
        n_resamples (int): Number of bootstrap resamples (default 1000)
        ci (float): Confidence interval level (default 0.95)
        seed (int): Random seed

    Returns:
        dict: Mean and CI bounds for each metric
    """
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)

    bootstrap_metrics = {
        "accuracy": [], "recall": [], "specificity": [], "f1_score": [], "auc_roc": []
    }

    for _ in range(n_resamples):
        indices = rng.choice(n, n, replace=True)
        bt = y_true[indices]
        bp = y_pred[indices]
        bprobs = y_probs[indices]

        if len(np.unique(bt)) < 2:
            continue

        m = compute_binary_metrics(bt, bp, bprobs)
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(m[key])

    alpha = (1 - ci) / 2
    results = {}
    for key, values in bootstrap_metrics.items():
        values = np.array(values)
        results[key] = {
            "mean":  round(float(np.mean(values)), 4),
            "lower": round(float(np.percentile(values, alpha * 100)), 4),
            "upper": round(float(np.percentile(values, (1 - alpha) * 100)), 4),
        }
    return results


# ─────────────────────────────────────────────
# Training Loop Helpers
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one training epoch.

    Args:
        model: PyTorch model
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: CUDA or CPU

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Handle BloodMNIST labels (may be 2D)
        if labels.dim() > 1:
            labels = labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        logits  = outputs.logits
        loss    = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device, binary=True):
    """
    Evaluate model on a given DataLoader.

    Args:
        model: PyTorch model
        loader: Val or Test DataLoader
        criterion: Loss function
        device: CUDA or CPU
        binary (bool): True for binary, False for multi-class

    Returns:
        dict: loss, accuracy, f1, predictions, labels, probabilities
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if labels.dim() > 1:
                labels = labels.squeeze(1)

            outputs = model(images)
            logits  = outputs.logits
            loss    = criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)

    if binary:
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
        probs_pos = [p[1] for p in all_probs]
    else:
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        probs_pos = all_probs

    return {
        "loss":   avg_loss,
        "acc":    acc,
        "f1":     f1,
        "preds":  all_preds,
        "labels": all_labels,
        "probs":  probs_pos,
    }


# ─────────────────────────────────────────────
# Saving Results
# ─────────────────────────────────────────────
def save_metrics(metrics, filepath):
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {filepath}")


def save_training_log(log_data, filepath):
    """
    Save epoch-level training statistics to CSV.

    Args:
        log_data (list of dict): Each dict = one epoch's stats
        filepath (str): Output CSV path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(log_data)
    df.to_csv(filepath, index=False)
    print(f"Training log saved to: {filepath}")


def set_seed(seed=42):
    """Fix random seed for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Random seed fixed: {seed}")
