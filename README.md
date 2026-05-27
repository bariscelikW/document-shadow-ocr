# OCR-Aware Document Shadow Removal

Document shadow removal is commonly optimized using pixel-level reconstruction losses such as L1, SSIM, or PSNR. However, these metrics do not directly measure whether the restored document becomes more readable for OCR systems.

This project explores whether OCR-aware supervision improves document shadow removal quality beyond traditional pixel reconstruction objectives.

---

## Project Goal

> Does optimizing for OCR-related objectives improve document readability better than optimizing only for pixel similarity?

Instead of evaluating only visual similarity, this project also evaluates OCR recognition performance using CER and WER measured by Tesseract — a different engine from the one used during training, to avoid evaluation circularity.

---

## Results (SD7K Test Set)

| Method | PSNR | SSIM | CER ↓ | WER ↓ |
|---|---|---|---|---|
| pixel | 14.58 | 0.844 | 0.593 | 0.777 |
| edge_only | **24.86** | **0.951** | **0.249** | **0.408** |
| edge (α=0.5) | 14.96 | 0.878 | 0.359 | 0.543 |
| ocr_feature | 14.67 | 0.857 | 0.589 | 0.744 |

*FSENet (SOTA): PSNR=28.67, SSIM=0.960 — reported for reference only, not reproduced here.*

**Key finding:** `edge_only` achieves 2.5× better OCR accuracy than the pixel baseline (CER 0.249 vs 0.593), confirming that high PSNR does not guarantee OCR readability.

---

## Experiments

| Experiment | Loss | Description |
|---|---|---|
| `pixel` | L1 + SSIM | Baseline pixel reconstruction |
| `edge_only` | L_edge | Sobel gradient loss inside text regions only |
| `edge` | L_pixel + 0.5 × L_edge | Pixel + edge supervision (α=0.5) |
| `ocr_feature` | L_pixel + 50 × L_ocr | Pixel + CRNN feature consistency |
| `edge_v2` | L_pixel + 2.0 × L_edge | Pixel + edge supervision with stronger edge weight |
| `triple_feat` | L_pixel + 2.0 × L_edge + 25 × L_ocr | All three combined |
| `ocr_ctc_direct` | L_ctc | Direct OCR supervision via CTC loss (experimental) |

---

## Method

### Architecture

- Lightweight U-Net (~2M parameters) with residual output formulation
- Input: 256×256 random crops during training
- Inference: full-resolution tiled inference (256×256 tiles, 32-pixel overlap)

### Text Mask Pipeline

Text regions are detected using PaddleOCR on clean (shadow-free) ground truth images, then expanded with 8-pixel dilation and Gaussian blur (σ=3) to create soft binary masks. These masks restrict edge and OCR losses to text areas only.

### OCR Supervision

Two OCR-aware strategies were explored:

#### 1. OCR Feature Loss

Feature maps extracted from a frozen pretrained CRNN (ResNet backbone) are compared between the restored image and the ground truth. This encourages restoration of OCR-relevant texture without explicitly decoding text.

```
L_ocr = L1( F_crnn(pred) · mask, F_crnn(gt) · mask )
```

#### 2. Direct OCR CTC Loss (Experimental)

Text regions are cropped using bounding boxes from OCR annotations and passed through a full CRNN pipeline. CTC loss is computed directly against ground-truth text labels from JSON annotations.

```
L_ctc = CTC( CRNN(crop(pred, bbox)), gt_text )
```

This approach uses axis-aligned crops from OCR polygon bounding boxes. It was computationally expensive and less stable during training without pixel supervision.

### Evaluation Protocol

- **Pixel metrics:** PSNR, SSIM, RMSE on full images
- **Region metrics:** PSNR computed separately for text and non-text regions
- **OCR metrics:** CER and WER using Tesseract (`--psm 6 --oem 3`)
- **Training detection:** PaddleOCR — **Evaluation:** Tesseract (different engine, avoids circularity)

---

## Datasets

| Dataset | Split | Images | Notes |
|---|---|---|---|
| SD7K | train | 2,678 | Non-Latin images excluded |
| SD7K | val | 399 | Used for hyperparameter selection |
| SD7K | test | 341 | Main evaluation |
| Kligler | test | 300 | Cross-dataset generalization |

```
splits/
├── train/
│   ├── input/        # shadowed images
│   ├── target/       # clean ground truth
│   └── ocr_gt/       # PaddleOCR annotations (.json) + text masks (.png)
├── val/
├── test_sd7k/
├── test_kliglers/
└── test_jungs/
```

---

## Training

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 4×10⁻⁴ |
| LR scheduler | Cosine Annealing |
| Input resolution | 256×256 random crops |
| Augmentation | Random horizontal flip + small affine |
| Mixed precision | AMP enabled |
| Environment | Google Colab A100 GPU |

---

## Requirements

```
Python 3.10+
PyTorch 2.x
albumentations
pytesseract
editdistance
deep-text-recognition-benchmark  (submodule, for CRNN)
```

Install Tesseract:
```bash
apt-get install -q tesseract-ocr
pip install pytesseract editdistance albumentations
```

---

## References

- Li et al., FSENet, ICCV 2023
- Lin et al., BEDSR-Net, CVPR 2020
- Yang et al., DocDiff, ACM MM 2023
- Qin et al., DENet, ACCV 2022
- Yin et al., PE-YOLO, ICANN 2023
- Matsuo & Aoki, Sensors 2024
