"""
inference.py — Run Inference on New X-Ray Images
==================================================
Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
Authors: Munaza Tariq, Amaim Anwar, Areeba Arshad | FAST-NUCES

Usage:
    python inference.py --image_path chest_xray.jpg --phase 3
    python inference.py --image_path chest_xray.jpg --phase 2
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model


# Class labels
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ImageNet normalization
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
MEAN = processor.image_mean
STD  = processor.image_std


def load_image(image_path):
    """
    Load and preprocess a chest X-ray image for inference.

    Args:
        image_path (str): Path to the image file

    Returns:
        tensor: Preprocessed image tensor (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def predict(image_path, phase=3, checkpoint_path=None, device=None):
    """
    Run inference on a single chest X-ray image.

    Args:
        image_path (str): Path to the X-ray image
        phase (int): Model phase (2 or 3)
        checkpoint_path (str): Path to model weights
        device: CUDA or CPU

    Returns:
        dict: prediction, confidence, probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/phase{phase}_kermany_best.pth"

    # Load model
    print(f"Loading Phase {phase} model from: {checkpoint_path}")
    model = get_model(phase=phase, num_labels=2)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Model weights loaded successfully")
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print("   Running with random weights (for demo only)")

    model = model.to(device)
    model.eval()

    # Preprocess image
    image_tensor = load_image(image_path).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1)[0]
        pred    = torch.argmax(probs).item()

    result = {
        "prediction":       CLASS_NAMES[pred],
        "confidence":       round(probs[pred].item() * 100, 2),
        "prob_normal":      round(probs[0].item() * 100, 2),
        "prob_pneumonia":   round(probs[1].item() * 100, 2),
    }

    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image:  {args.image_path}")
    print(f"Phase:  {args.phase}")
    print("-" * 40)

    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return

    result = predict(
        image_path      = args.image_path,
        phase           = args.phase,
        checkpoint_path = args.checkpoint,
        device          = device
    )

    print("\n" + "=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)
    print(f"  Prediction:   {result['prediction']}")
    print(f"  Confidence:   {result['confidence']}%")
    print(f"  P(NORMAL):    {result['prob_normal']}%")
    print(f"  P(PNEUMONIA): {result['prob_pneumonia']}%")
    print("=" * 40)

    if result["prediction"] == "PNEUMONIA":
        print("⚠️  PNEUMONIA DETECTED — Please consult a physician.")
    else:
        print("✅ No pneumonia detected — Lung appears NORMAL.")

    print("\nNote: This is an AI screening tool, not a clinical diagnosis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ViT Pneumonia Detection Inference"
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to chest X-ray image"
    )
    parser.add_argument(
        "--phase", type=int, default=3, choices=[2, 3],
        help="Model phase to use for inference (default: 3)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (optional)"
    )
    args = parser.parse_args()
    main(args)
