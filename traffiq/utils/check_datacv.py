"""
============================================================
 TRAFFIQ v2 — Utilities
 File: utils/tools_v2.py

 Contains two tools:

 1. migrate_labels  — converts old v1 labels.json
    (which had 'steering') to v2 format
    (which needs 'direction' and 'speed')

 2. check_dataset_v2 — validates your v2 dataset
    before training. Checks both speed and direction
    distributions.

 Usage:
   # Migrate old labels
   python3 utils/tools_v2.py migrate --data_dir ./dataset/OLD

   # Check v2 dataset
   python3 utils/tools_v2.py check --data_dir ./dataset/NEW
============================================================
"""

import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# ─── MIGRATE v1 → v2 LABELS ───────────────────────────────

def migrate_labels(data_dir: str):
    """
    Converts v1 labels.json format:
      {"image_path": "...", "steering": 0.3, "throttle": 0.4}

    To v2 format:
      {"image_path": "...", "direction": 0.3, "speed": 0.4}

    The old 'steering' field becomes 'direction'.
    The old 'throttle' field becomes 'speed'.
    If 'throttle' was missing (hardcoded in v1), defaults to 0.3.
    """
    label_file = Path(data_dir) / "labels.json"
    if not label_file.exists():
        print(f"[Error] labels.json not found in {data_dir}")
        return

    with open(label_file) as f:
        records = json.load(f)

    # Check if already v2 format
    if "direction" in records[0]:
        print("[Info] Labels already in v2 format. Nothing to do.")
        return

    migrated = []
    for rec in records:
        migrated.append({
            "image_path": rec["image_path"],
            "direction":  rec.get("steering", 0.0),
            "speed":      rec.get("throttle", 0.3),   # default 0.3 if missing
            "timestamp":  rec.get("timestamp", 0)
        })

    # Save backup of original
    backup_path = Path(data_dir) / "labels_v1_backup.json"
    with open(backup_path, "w") as f:
        json.dump(records, f)

    # Write v2 labels
    with open(label_file, "w") as f:
        json.dump(migrated, f, indent=2)

    print(f"[Migrated] {len(migrated)} records converted to v2 format.")
    print(f"[Backup]   Original saved to {backup_path}")
    print(f"\n[Warning] v1 data had hardcoded throttle.")
    print(f"          Speed labels will be uniform (~0.3).")
    print(f"          The model will learn poor speed control.")
    print(f"          Collect fresh v2 data for best results.")


# ─── CHECK v2 DATASET ─────────────────────────────────────

def check_dataset_v2(data_dir: str):
    label_file = Path(data_dir) / "labels.json"
    if not label_file.exists():
        print(f"[Error] labels.json not found in {data_dir}")
        return

    with open(label_file) as f:
        records = json.load(f)

    total = len(records)
    print("\n" + "="*55)
    print("  TRAFFIQ v2 Dataset Quality Report")
    print("="*55)

    # ── Frame count ──────────────────────────────────────
    print(f"\n[1] Frame Count: {total}  ", end="")
    print("✓" if total >= 5000 else f"✗ (need {5000-total} more)")

    # ── Format check ─────────────────────────────────────
    sample = records[0]
    has_direction = "direction" in sample
    has_speed     = "speed" in sample
    print(f"\n[2] Label format:")
    print(f"    'direction' field: {'✓' if has_direction else '✗ MISSING — run migrate'}")
    print(f"    'speed' field:     {'✓' if has_speed     else '✗ MISSING — run migrate'}")

    if not (has_direction and has_speed):
        print("\n[!] Fix label format before continuing.")
        return

    # ── Image integrity ───────────────────────────────────
    print(f"\n[3] Checking images (sample of 200)...")
    sample_recs = records[:200]
    bad = [r for r in sample_recs if cv2.imread(r["image_path"]) is None]
    print(f"    {'✓ All readable' if not bad else f'✗ {len(bad)} unreadable'}")

    # ── Direction distribution ────────────────────────────
    directions = [r["direction"] for r in records]
    straight   = sum(1 for d in directions if abs(d) < 0.05)
    turning    = total - straight
    pct_str    = straight / total * 100

    print(f"\n[4] Direction distribution:")
    print(f"    Straight: {straight} ({pct_str:.0f}%)  ", end="")
    print("✓" if pct_str <= 65 else "⚠ Too many straight frames")
    print(f"    Turning:  {turning} ({100-pct_str:.0f}%)")

    # ── Speed distribution ────────────────────────────────
    speeds     = [r["speed"] for r in records]
    uniform    = np.std(speeds) < 0.02   # std < 0.02 means nearly uniform
    mean_speed = np.mean(speeds)
    std_speed  = np.std(speeds)

    print(f"\n[5] Speed distribution:")
    print(f"    Mean: {mean_speed:.3f}  Std: {std_speed:.3f}  ", end="")
    if uniform:
        print("⚠ Speed is nearly uniform (was throttle hardcoded?)")
        print("    The model will NOT learn proper speed control.")
        print("    Recollect with collect_data_v2.py to vary speed.")
    else:
        print("✓ Good variation")

    near_zero  = sum(1 for s in speeds if s < 0.05)
    fast       = sum(1 for s in speeds if s > 0.4)
    print(f"    Near-stop frames: {near_zero} ({near_zero/total*100:.0f}%)")
    print(f"    Fast frames (>0.4): {fast} ({fast/total*100:.0f}%)")

    # ── Plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("TRAFFIQ v2 Dataset Distribution", fontsize=13)

    axes[0].hist(directions, bins=50, color="#4A90D9", edgecolor="white")
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title("Direction labels")
    axes[0].set_xlabel("Direction [-1, 1]")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(speeds, bins=30, color="#2ECC71", edgecolor="white")
    axes[1].set_title("Speed labels")
    axes[1].set_xlabel("Speed [-1, 1]")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(data_dir) / "dataset_distribution_v2.png"
    plt.savefig(out, dpi=120)
    print(f"\n[Saved] Distribution plot → {out}")
    plt.show()

    # ── Summary ──────────────────────────────────────────
    ready = (
        total >= 5000
        and has_direction and has_speed
        and not bad
        and pct_str <= 65
        and not uniform
    )
    print("\n" + "="*55)
    if ready:
        print("  ✓ Dataset ready for training.")
        print(f"  → python3 training/train_v2.py --data_dir {data_dir}")
    else:
        print("  ✗ Issues found — fix before training.")
    print()


# ─── ENTRY POINT ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAFFIQ v2 Dataset Tools")
    sub    = parser.add_subparsers(dest="command")

    # migrate subcommand
    p_migrate = sub.add_parser("migrate", help="Convert v1 labels to v2")
    p_migrate.add_argument("--data_dir", required=True)

    # check subcommand
    p_check = sub.add_parser("check", help="Validate v2 dataset")
    p_check.add_argument("--data_dir", required=True)

    args = parser.parse_args()

    if args.command == "migrate":
        migrate_labels(args.data_dir)
    elif args.command == "check":
        check_dataset_v2(args.data_dir)
    else:
        parser.print_help()