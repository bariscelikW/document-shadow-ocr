"""
generate_ocr_gt.py
------------------
Runs PaddleOCR on the clean (shadow-free) images of SD7K and saves:
  - Per-image JSON with detected text boxes, transcriptions, and confidence scores
  - Per-image PNG soft text-region mask (binary → dilated → Gaussian-blurred)

Speed optimisations
-------------------
1. Deduplication: many shadowed inputs share the same clean target (same
   document photographed under different shadows). We hash every target image
   (MD5 of raw bytes) and run OCR only once per unique file, then copy the
   JSON + mask to all duplicates. This typically reduces work by 3-5x on SD7K.

2. Disabled heavy preprocessing: UVDoc (document unwarping) and orientation
   classification are skipped because SD7K targets are already flat/clean.
   This alone cuts per-image time roughly in half.

Compatible with PaddleOCR 3.x.

Usage
-----
  python scripts/generate_ocr_gt.py --sd7k_root "D:/datasets/SD7K"

Resumable: images whose JSON+mask already exist are skipped.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.65
DILATION_PX = 8
GAUSS_SIGMA = 3.0
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def file_md5(path: Path) -> str:
    """MD5 of raw file bytes — fast enough for ~10 MB PNGs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_duplicate_map(image_paths: list) -> dict:
    """
    Returns {img_path: canonical_path} for every image.
    If an image is unique its canonical is itself.
    If it is a duplicate its canonical is the first occurrence with that hash.
    Also prints a summary of how many unique images were found.
    """
    hash_to_first: dict = {}
    canon: dict = {}
    for p in image_paths:
        h = file_md5(p)
        if h not in hash_to_first:
            hash_to_first[h] = p
        canon[p] = hash_to_first[h]

    n_unique = len(hash_to_first)
    n_total = len(image_paths)
    print(
        f"  Dedup: {n_unique} unique / {n_total} total "
        f"({n_total - n_unique} duplicates will be copied, not re-run)"
    )
    return canon


def parse_ocr_result(result) -> list:
    """
    Parse PaddleOCR 3.x predict() result into a flat list of dicts.
    OCRResult is a dict subclass — use dict access, not attribute access.
    """
    detections = []
    if not result:
        return detections
    for page in result:
        if page is None:
            continue
        if isinstance(page, dict) and "rec_polys" in page:
            polys = page.get("rec_polys") or []
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            for poly, text, score in zip(polys, texts, scores):
                bbox_int = [[int(poly[i][0]), int(poly[i][1])] for i in range(4)]
                detections.append(
                    {"bbox": bbox_int, "text": text, "confidence": float(score)}
                )
            continue
        # Fallback: 2.x-style (box, (text, score)) tuples
        try:
            for line in page:
                if line is None:
                    continue
                if isinstance(line, (list, tuple)) and len(line) == 2:
                    box_part, rec_part = line
                    if isinstance(rec_part, (list, tuple)) and len(rec_part) == 2:
                        text, score = rec_part
                        bbox_int = [[int(pt[0]), int(pt[1])] for pt in box_part]
                        detections.append(
                            {
                                "bbox": bbox_int,
                                "text": str(text),
                                "confidence": float(score),
                            }
                        )
        except TypeError:
            pass
    return detections


def build_mask(
    h: int, w: int, detections: list, dilation: int, sigma: float
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        pts = np.array(det["bbox"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1)
    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    ksize = int(6 * sigma + 1) | 1
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigmaX=sigma)
    return np.clip(mask_f, 0.0, 1.0)


def run_ocr_and_save(
    ocr,
    img_path: Path,
    json_out: Path,
    mask_out: Path,
    conf_threshold: float,
    dilation: int,
    sigma: float,
) -> bool:
    """Run OCR on img_path, write json_out and mask_out. Returns True on success."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"\n[ERROR] Cannot read: {img_path}")
        return False
    h, w = img_bgr.shape[:2]

    try:
        result = ocr.predict(img_bgr)
    except Exception as exc:
        print(f"\n[ERROR] OCR failed on {img_path.name}: {exc}")
        return False

    all_dets = parse_ocr_result(result)
    kept_dets = [d for d in all_dets if d["confidence"] >= conf_threshold]

    mask_f = build_mask(h, w, kept_dets, dilation, sigma)
    mask_u8 = (mask_f * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(mask_out, optimize=True)

    record = {
        "image_file": str(img_path.relative_to(img_path.parents[2])),
        "width": w,
        "height": h,
        "detections": kept_dets,
        "n_kept": len(kept_dets),
        "n_total": len(all_dets),
        "conf_threshold": conf_threshold,
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return True


def copy_outputs(
    src_json: Path, src_mask: Path, dst_json: Path, dst_mask: Path, new_image_file: str
):
    """Copy JSON (with updated image_file field) and mask to a duplicate target."""
    with open(src_json, encoding="utf-8") as f:
        record = json.load(f)
    record["image_file"] = new_image_file
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    shutil.copy2(src_mask, dst_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Per-split processing
# ──────────────────────────────────────────────────────────────────────────────


def process_split(
    ocr, split_name, target_dir, out_dir, conf_threshold, dilation, sigma
):
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in target_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        print(f"[WARNING] No images found in {target_dir}")
        return {}

    print(f"  Hashing {len(image_paths)} images for deduplication ...")
    canon_map = build_duplicate_map(image_paths)

    stats = {
        "total_images": len(image_paths),
        "skipped": 0,
        "processed": 0,
        "copied": 0,
        "errors": 0,
    }

    for img_path in tqdm(image_paths, desc=f"  {split_name}", unit="img", ncols=90):
        json_out = out_dir / (img_path.stem + ".json")
        mask_out = out_dir / (img_path.stem + "_mask.png")

        # Already done
        if json_out.exists() and mask_out.exists():
            stats["skipped"] += 1
            continue

        canon = canon_map[img_path]

        if canon == img_path:
            # This is a unique image — run OCR
            ok = run_ocr_and_save(
                ocr, img_path, json_out, mask_out, conf_threshold, dilation, sigma
            )
            if ok:
                stats["processed"] += 1
            else:
                stats["errors"] += 1
        else:
            # Duplicate — wait for canonical to be done, then copy
            canon_json = out_dir / (canon.stem + ".json")
            canon_mask = out_dir / (canon.stem + "_mask.png")

            if not (canon_json.exists() and canon_mask.exists()):
                # Canonical hasn't been processed yet in this run;
                # it will appear later in the sorted list — skip for now.
                # The resume logic will catch it on the next run if needed.
                # Better: process canonical first (it appears earlier alphabetically
                # if stems are sorted, which they are).
                # If somehow it's not ready, just run OCR directly.
                ok = run_ocr_and_save(
                    ocr, img_path, json_out, mask_out, conf_threshold, dilation, sigma
                )
                if ok:
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
                continue

            new_image_file = str(img_path.relative_to(img_path.parents[2]))
            try:
                copy_outputs(canon_json, canon_mask, json_out, mask_out, new_image_file)
                stats["copied"] += 1
            except Exception as exc:
                print(f"\n[ERROR] Copy failed for {img_path.name}: {exc}")
                stats["errors"] += 1

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate PaddleOCR GT + masks for SD7K (3.x, fast)."
    )
    parser.add_argument("--sd7k_root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--target_subdir", default="target")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--dilation", type=int, default=DILATION_PX)
    parser.add_argument("--sigma", type=float, default=GAUSS_SIGMA)
    args = parser.parse_args()

    sd7k_root = Path(args.sd7k_root).resolve()
    if not sd7k_root.exists():
        raise FileNotFoundError(f"SD7K root not found: {sd7k_root}")

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("Run: pip install paddleocr")

    print("Initialising PaddleOCR 3.x on CPU ...")
    print(
        "  (UVDoc unwarping + orientation classify disabled — not needed for flat docs)"
    )
    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        use_doc_orientation_classify=False,  # skip orientation model
        use_doc_unwarping=False,  # skip UVDoc — biggest speedup
    )
    print(f"  Confidence threshold : {args.conf}")
    print(f"  Mask dilation        : {args.dilation} px")
    print(f"  Gaussian sigma       : {args.sigma}\n")

    out_root = sd7k_root / "ocr_gt"
    all_stats = {}

    for split in args.splits:
        target_dir = sd7k_root / split / args.target_subdir
        if not target_dir.exists():
            print(f"[WARNING] Skipping '{split}': not found at {target_dir}")
            continue
        print(f"Processing split: {split}  ({target_dir})")
        all_stats[split] = process_split(
            ocr,
            split,
            target_dir,
            out_root / split,
            args.conf,
            args.dilation,
            args.sigma,
        )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for split, s in all_stats.items():
        print(
            f"  {split:6s}  total={s.get('total_images', 0):5d}  "
            f"processed={s.get('processed', 0):5d}  "
            f"copied={s.get('copied', 0):5d}  "
            f"skipped={s.get('skipped', 0):5d}  "
            f"errors={s.get('errors', 0):3d}"
        )
    print(f"\nOutputs written to: {out_root}")


if __name__ == "__main__":
    main()
