# LAB_ASSIGNMENT
# Vision Transformer (ViT) on CIFAR-10

## How to run in Colab

1. Open `q1.ipynb` in Google Colab.
2. Set Runtime → Change runtime type → GPU.
3. Run all cells top-to-bottom.
4. After training, the **best test accuracy** is printed and saved at `/content/best_accuracy.txt`.

## Config for Best Model

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Batch Size | 128 |
| Epochs | 80 |
| Learning Rate | 3e-4 |
| Weight Decay | 0.05 |
| Image Size | 32 |
| Patch Size | 4 |
| Embedding Dim | 256 |
| Depth | 8 |
| Num Heads | 8 |
| MLP Ratio | 4.0 |
| Dropout | 0.0 |
| Drop Path Rate | 0.1 |
| MixUp Alpha | 0.8 |
| Label Smoothing | 0.1 |
| RandAugment | Enabled |
| Gradient Clipping | 1.0 |

## Results (CIFAR-10)

| Model | Test Accuracy (%) |
|-------|-------------------|
| ViT(Scratch) | 82.67 |

## Short Analysis

- **Patch size choices**: 4×4 patches balance accuracy vs memory; smaller patches improve performance but increase computation.
  
- **Depth/width trade-off**: 8 layers,256-dim embedding is a good baseline; increasing depth or embedding dimension can boost accuracy at cost of memory.
  
- **Augmentations**: RandAugment+MixUp improves generalization significantly.
  
- **Optimizer & Schedule**: AdamW + cosine LR with warmup stabilizes training and speeds up convergence.
  
- **CLS token classification**: Extracting from first token works well for image classification.
    
- **Optional improvements**: Pretrained ViT, CutMix, or AutoAugment can further improve results with fewer epochs.

---

# Q2 — Text-Driven Image Segmentation (SAM 2)

## How to run in Colab

1. Open `q2.ipynb`.
2. Set Runtime → GPU.
3. Run all cells to perform text-prompted segmentation.

## Pipeline

1. Load image.  
2. Accept text prompt for object selection.  
3. Convert text → region seeds (via CLIPSeg / GroundingDINO / GLIP).  
4. Feed seeds into **SAM 2** to generate mask.  
5. Display final mask overlay.

## Notes / Limitations

- Works for **single images**.  
- Mask quality depends on prompt specificity.  
- Optional video extension can propagate masks across frames (bonus).

---

**Submission Instructions**

- Both notebooks must execute top-to-bottom in Colab GPU.  
- Repo should contain only: `q1.ipynb`, `q2.ipynb`, `README.md`.  
- Submit your **best test accuracy** (82.67%) via Google Form with repo link.
