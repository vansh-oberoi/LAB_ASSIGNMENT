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

#Text-Driven Image Segmentation (SAM 2)

How to run in Colab
-------------------
1. Open `q2.ipynb` in Google Colab.
2. Set Runtime → Change runtime type → GPU.
3. Run all cells top-to-bottom.
4. Use either the sample demo image or upload your own image and enter a text prompt.

Pipeline
--------
Step                     Description
------------------------------------------------------------
Load Image               Load from URL or upload from local storage.
Text → Bounding Boxes    Uses OwlViT (zero-shot object detection) to detect regions based on text prompt.
Bounding Boxes → Masks   Refined by SAM 2 (Segment Anything v2, Hiera Large checkpoint) to generate segmentation masks.
Visualization            Overlays masks with color blending; displays original image, bounding boxes, and mask overlay.
Multi-Object Segmentation Accepts multiple text prompts (e.g., "person", "dog", "car") and assigns distinct colors per class.

Notes / Limitations
-------------------
- Works for single images (no video support).
- Mask quality depends on prompt specificity and OwlViT detection accuracy.
- Supports multiple objects, but overlapping regions may merge.
- Requires GPU runtime for practical speed.

Short Analysis
--------------
- Text-to-box step: OwlViT allows segmentation without manual clicks, enabling text-driven interaction.
- SAM 2 refinement: Produces sharper and more precise masks from bounding boxes.
- Multi-object extension: Demonstrates scalability with prompts like "dog" + "car" in the same scene.
- Limitations: Dependent on text-prompt accuracy; may miss small/ambiguous objects.
- Future improvements: Add video mask propagation, use stronger grounding models (e.g., GroundingDINO).
