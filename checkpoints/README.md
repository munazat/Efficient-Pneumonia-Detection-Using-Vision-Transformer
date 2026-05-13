# Model Checkpoints

This directory stores trained model weights.

## Files Expected After Training

| File | Description |
|------|-------------|
| `phase2_kermany_best.pth` | Phase 2 best checkpoint (Full Fine-Tuning) |
| `phase3_kermany_best.pth` | Phase 3 best checkpoint (Partial Fine-Tuning) |

## Why Weights Are Not Included

Model weights (~330MB each) are too large for GitHub.

## How to Get the Weights

### Option 1 — Train from scratch
```bash
python train.py --phase 2 --dataset kermany
python train.py --phase 3 --dataset kermany
```

### Option 2 — Download pretrained weights
If provided separately by the authors, place them here:
```
checkpoints/
├── phase2_kermany_best.pth
└── phase3_kermany_best.pth
```

## Model Architecture

Both checkpoints use ViT-Base-Patch16-224 base architecture.

### Phase 2 Checkpoint
- All 86M parameters trained
- Best Val F1: 0.9760 (Epoch 1)
- Test AUC-ROC: 0.97

### Phase 3 Checkpoint
- Only 14M parameters trained (84% frozen)
- Best Val F1: 0.9527 (Epoch 5)
- Test Recall: 94.87%
- Test AUC-ROC: 0.9487
