"""
============================================================
 TRAFFIQ — Dataset Quality Checker
 Run this BEFORE training to validate your collected data.

 Checks:
   ✓ Minimum frame count (5000)
   ✓ Steering distribution (not too biased toward 0)
   ✓ No corrupted/unreadable images
   ✓ Image resolution consistency
   ✓ Class balance (turning vs straight)

 Usage:
   python3 utils/check_dataset.py --data_dir ./dataset/YOUR_SESSION
============================================================
"""

import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def check_dataset(data_dir: str):
    data_path = Path(data_dir)
    label_file = data_path / "labels.json"

    print("\n" + "="*55)
    print("  TRAFFIQ Dataset Quality Report")
    print("="*55)

    # ── Load labels ──────────────────────────────────────
    if not label_file.exists():
        print(f"[✗] labels.json not found in {data_dir}")
        return

    with open(label_file) as f:
        records = json.load(f)

    total = len(records)
    print(f"\n[1] Frame Count: {total}  ", end="")
    if total >= 5000:
        print(f"✓  (target: 5000)")
    else:
        print(f"✗  ({5000 - total} more needed!)")

    # ── Image integrity check ─────────────────────────────
    print(f"\n[2] Checking image files...")
    bad, sizes = [], []
    for rec in tqdm(records, desc="  Scanning", unit="img"):
        path = rec["image_path"]
        img = cv2.imread(path)
        if img is None:
            bad.append(path)
        else:
            sizes.append(img.shape)

    if bad:
        print(f"  ✗ {len(bad)} corrupted/missing images!")
        for b in bad[:5]:
            print(f"    → {b}")
    else:
        print(f"  ✓ All {total} images readable.")

    # ── Resolution consistency ────────────────────────────
    print(f"\n[3] Image resolutions:")
    size_counts = Counter(str(s) for s in sizes)
    for res, count in size_counts.most_common(5):
        print(f"  {res}: {count} images")
    if len(size_counts) > 1:
        print("  ⚠ Mixed resolutions detected — run preprocess during training.")

    # ── Steering distribution ─────────────────────────────
    steerings = [r["steering"] for r in records]
    straight  = sum(1 for s in steerings if abs(s) < 0.05)
    left      = sum(1 for s in steerings if s < -0.05)
    right     = sum(1 for s in steerings if s > 0.05)
    pct_str   = straight / total * 100

    print(f"\n[4] Steering Distribution:")
    print(f"  Straight (|s|<0.05): {straight:5d}  ({pct_str:.1f}%)", end="")
    if pct_str > 60:
        print("  ⚠ Heavily biased toward straight! Drive more turns.")
    else:
        print("  ✓")
    print(f"  Left turn (s<-0.05): {left:5d}  ({left/total*100:.1f}%)")
    print(f"  Right turn (s>0.05): {right:5d}  ({right/total*100:.1f}%)")

    # ── Plot histogram ────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.hist(steerings, bins=50, color="#4A90D9", edgecolor="white", alpha=0.85)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Center')
    plt.title("TRAFFIQ — Steering Angle Distribution", fontsize=13)
    plt.xlabel("Steering Angle")
    plt.ylabel("Frame Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = data_path / "steering_distribution.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n[Saved] Histogram → {out_path}")
    plt.show()

    # ── Summary ───────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    ready = (total >= 5000 and len(bad) == 0 and pct_str <= 70)
    if ready:
        print("  ✓ Dataset looks good. Ready to train!")
        print(f"  → Run: python3 training/train_dave2.py --data_dir {data_dir}")
    else:
        print("  ✗ Issues found. Fix them before training.")
        if total < 5000:
            print(f"    • Collect {5000 - total} more frames.")
        if pct_str > 70:
            print("    • Drive more L-turns and curves.")
        if bad:
            print(f"    • Remove/re-collect {len(bad)} corrupted images.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAFFIQ Dataset Checker")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to your dataset folder")
    args = parser.parse_args()
    check_dataset(args.data_dir)