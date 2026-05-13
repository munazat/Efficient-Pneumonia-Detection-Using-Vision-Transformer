# Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection: A Partial Freezing Approach

**Course:** Deep Learning (CS-4112) | BS Data Science  
**Institution:** FAST-NUCES, Islamabad  
**Authors:** Munaza Tariq (i232545) | Amaim Anwar (i232614) | Areeba Arshad (i232656)

---

## 📌 Project Overview

This project presents a complete deep learning pipeline for automated pneumonia detection from chest X-ray images using a Vision Transformer (ViT) architecture. The work is conducted in three phases:

- **Phase 1:** Research Proposal — identified ViT-Base-Patch16-224 as the target architecture
- **Phase 2:** Reproduction of Singh et al. (2024) baseline with 5 improvements
- **Phase 3:** Proposed improvements using layer freezing, label smoothing, and validation bug fix

### Key Results

| Model | Accuracy | Recall | Specificity | F1-Score | AUC-ROC |
|-------|----------|--------|-------------|----------|---------|
| Singh et al. 2024 (Paper) | 97.61% | 95.00% | 98.00% | N/A | 0.96 |
| Phase 2 (Reproduced Baseline) | 91.67% | 92.56% | 90.17% | 93.28% | 0.97 |
| Phase 3 (Proposed Method) | 89.90% | **94.87%** | 81.62% | 92.15% | 0.9487 |
| BloodMNIST (Cross-Domain) | 90.30% | 88.82% | — | 89.25% | **0.9917** |

---

## 🗂️ Repository Structure

```
project-root/
│── README.md               # Project documentation
│── requirements.txt        # Python dependencies
│── train.py                # Main training script
│── inference.py            # Run inference on new images
│── config.yaml             # All hyperparameters and settings
│
├── data/
│   └── sample_data.csv     # 5-10 sample entries for demo
│
├── notebooks/
│   └── 01_inference_demo.ipynb  # Demo notebook
│
├── src/
│   ├── model.py            # ViT model definition and freezing
│   ├── dataset.py          # Dataset loading and preprocessing
│   └── utils.py            # Metrics, early stopping, utilities
│
├── results/
│   ├── baseline_metrics.json    # Phase 2 results
│   ├── improved_metrics.json    # Phase 3 results
│   └── training_log.csv         # Epoch-level training statistics
│
└── checkpoints/
    └── README.md           # Instructions to download model weights
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/vit-pneumonia-detection.git
cd vit-pneumonia-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the Kermany Chest X-Ray dataset from Kaggle:
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```
Place it in the following structure:
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### 4. Train the Model
```bash
# Train Phase 2 (Full Fine-Tuning Baseline)
python train.py --phase 2

# Train Phase 3 (Partial Fine-Tuning - Proposed Method)
python train.py --phase 3
```

### 5. Run Inference
```bash
python inference.py --image_path path/to/xray.jpg --phase 3
```

---

## 🏗️ Model Architecture

- **Base Model:** ViT-Base-Patch16-224 (Google, pretrained on ImageNet)
- **Framework:** PyTorch + HuggingFace Transformers
- **Input:** 224×224 chest X-ray images (grayscale → 3-channel)
- **Output:** Binary classification (NORMAL / PNEUMONIA)

### Phase 3 Freezing Strategy
```
FROZEN (ImageNet pretrained weights preserved):
  ├── Patch Embedding Layer
  └── Encoder Blocks 0-9

TRAINABLE (adapted for pneumonia detection):
  ├── Encoder Block 10
  ├── Encoder Block 11
  ├── Final LayerNorm
  └── Classification Head (2 classes)

Parameter Reduction: 86M → 14M (84% reduction)
```

---

## ⚙️ Key Hyperparameters

| Parameter | Phase 2 | Phase 3 |
|-----------|---------|---------|
| Trainable Parameters | ~86M | ~14M |
| Optimizer | Adam (lr=1e-4) | Adam (lr=1e-4) |
| Loss Function | CrossEntropyLoss + class weights | CrossEntropyLoss + class weights + LS(ε=0.1) |
| Max Epochs | 30 | 12 |
| Early Stopping | Patience=5 | Patience=4 |
| Batch Size | 32 | 32 |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

---

## 📊 Datasets

### Kermany Chest X-Ray Dataset
- **Total Images:** 5,863 chest X-rays
- **Classes:** NORMAL (1,583) | PNEUMONIA (4,273)
- **Split:** Train 4,172 | Val 1,044 | Test 624
- **Source:** Kermany et al. (2018), available on Kaggle

### BloodMNIST (Cross-Domain Evaluation)
- **Total Images:** 17,092 blood cell microscopy images
- **Classes:** 8 cell types
- **Split:** Train 11,959 | Val 1,712 | Test 3,421
- **Source:** MedMNIST v2 (Yang et al., 2023)

---

## 📈 Our 3 Proposed Improvements (Phase 3)

1. **Layer Freezing:** Froze patch embedding + encoder blocks 0-9, reducing trainable parameters by 84% to prevent training collapse
2. **Label Smoothing (ε=0.1):** Replaced hard one-hot labels with soft labels to prevent overconfident predictions
3. **Validation Bug Fix:** Fixed data leakage where validation images were receiving training augmentations, corrupting early stopping signal

---

## 📚 References

1. Singh et al. (2024). *Efficient Pneumonia Detection Using Vision Transformers on Chest X-Rays.* Scientific Reports.
2. Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
3. Kermany et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell.
4. Yang et al. (2023). *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.* Scientific Data.

---

## 📝 License
This project is for academic purposes only. Dataset usage follows the respective dataset licenses.
