"""
============================================================
 TRAFFIQ — DAVE-2 CNN Training Script

 What this does:
   1. Loads your collected dataset
   2. Applies augmentation (brightness, flip, shadow)
   3. Trains DAVE-2 CNN for steering prediction
   4. Validates and plots loss curves
   5. Exports to TFLite (INT8) for Raspberry Pi

 Usage:
   python3 training/train_dave2.py --data_dir dataset/20260327_005825
   python3 training/train_dave2.py --data_dir dataset/20260327_005825 --epochs 50
============================================================
"""

import os
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ─── CONFIGURATION ────────────────────────────────────────
IMG_HEIGHT   = 66
IMG_WIDTH    = 200
IMG_CHANNELS = 3
BATCH_SIZE   = 64
EPOCHS       = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH  = "models/dave2_traffiq.h5"
TFLITE_SAVE_PATH = "models/dave2_traffiq.tflite"
# ──────────────────────────────────────────────────────────

# Project root: the traffiq/ directory (one level up from training/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── DATA AUGMENTATION ────────────────────────────────────

def augment_brightness(image: np.ndarray) -> np.ndarray:
    """Randomly brighten/darken to simulate variable lighting"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = 0.4 + np.random.uniform()   # 0.4 → 1.4
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_shadow(image: np.ndarray) -> np.ndarray:
    """Add a random shadow stripe — mimics real-world lighting drops"""
    h, w = image.shape[:2]
    x1, x2 = np.random.randint(0, w, 2)
    shadow_mask = np.zeros_like(image[:, :, 0])
    pts = np.array([[x1, 0], [x2, h], [w, h], [w, 0]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [pts], 1)
    image = image.copy().astype(np.float32)
    image[shadow_mask == 1] *= 0.5
    return np.clip(image, 0, 255).astype(np.uint8)


def augment_flip(image: np.ndarray, steering: float):
    """Mirror the image horizontally and negate steering"""
    return cv2.flip(image, 1), -steering


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Crop sky + hood, resize to DAVE-2 input, convert to YUV.
    Simulator images are 120x160. We crop top 20px and bottom 10px
    to remove irrelevant areas, keeping the road (rows 20-110).
    YUV helps the CNN focus on lane lines vs background.
    """
    # Crop: remove top 20px (sky/ceiling) and bottom 10px (car hood)
    # Input is 120x160, after crop → 90x160
    image = image[20:110, :, :]
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


# ─── DATASET LOADER ───────────────────────────────────────

class TraffiqDataset(tf.keras.utils.Sequence):
    def __init__(self, records: list, batch_size: int, augment: bool = False):
        self.records    = records
        self.batch_size = batch_size
        self.augment    = augment

    def __len__(self):
        return max(1, len(self.records) // self.batch_size)

    def __getitem__(self, idx):
        batch = self.records[idx * self.batch_size : (idx + 1) * self.batch_size]
        images, steerings = [], []

        for rec in batch:
            # Resolve image path relative to project root
            img_path = str(PROJECT_ROOT / rec["image_path"])
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            steering = float(rec["steering"])

            if self.augment:
                img = augment_brightness(img)
                if np.random.random() > 0.5:
                    img = augment_shadow(img)
                if np.random.random() > 0.5:
                    img, steering = augment_flip(img, steering)

            img = preprocess(img)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            steerings.append(steering)

        if len(images) == 0:
            # Return dummy batch to avoid crash
            return (np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32),
                    np.zeros((1,), dtype=np.float32))

        return np.array(images), np.array(steerings)

    def on_epoch_end(self):
        np.random.shuffle(self.records)


# ─── DAVE-2 MODEL ─────────────────────────────────────────

def build_dave2_model():
    """
    NVIDIA DAVE-2 architecture.
    Input:  (66, 200, 3) — YUV image
    Output: scalar steering angle in [-1, 1]
    """
    model = models.Sequential([
        # Normalization
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),

        # 5 Convolutional layers (feature extraction)
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),

        layers.Dropout(0.2),   # Prevent overfitting
        layers.Flatten(),

        # 4 Fully connected layers (decision making)
        layers.Dense(100, activation='elu'),
        layers.Dropout(0.2),
        layers.Dense(50,  activation='elu'),
        layers.Dense(10,  activation='elu'),

        # Output: single steering value
        layers.Dense(1, activation='tanh'),  # tanh keeps output in [-1, 1]
    ], name="DAVE2_TRAFFIQ")

    return model


# ─── TRAINING ─────────────────────────────────────────────

def train(data_dir: str, epochs: int):
    # Resolve data_dir relative to project root
    data_path = PROJECT_ROOT / data_dir
    label_file = data_path / "labels.json"

    if not label_file.exists():
        raise FileNotFoundError(f"labels.json not found in {data_path}")

    with open(label_file) as f:
        records = json.load(f)

    print(f"\n[Dataset] Loaded {len(records)} records from {data_path}")

    # Verify first image is accessible
    test_img_path = str(PROJECT_ROOT / records[0]["image_path"])
    test_img = cv2.imread(test_img_path)
    if test_img is None:
        raise FileNotFoundError(f"Could not read first image: {test_img_path}")
    print(f"[Dataset] Image shape: {test_img.shape} (H×W×C)")

    # Show steering distribution
    steerings = [r["steering"] for r in records]
    n_straight = sum(1 for s in steerings if abs(s) < 0.05)
    n_left  = sum(1 for s in steerings if s <= -0.05)
    n_right = sum(1 for s in steerings if s >= 0.05)
    print(f"[Dataset] Steering distribution: "
          f"Left={n_left} | Straight={n_straight} | Right={n_right}")

    # Balance dataset: reduce straight-driving bias
    straight = [r for r in records if abs(r["steering"]) < 0.05]
    turning  = [r for r in records if abs(r["steering"]) >= 0.05]

    if len(turning) > 0:
        max_straight = min(len(straight), len(turning) * 2)
        balanced = turning + straight[:max_straight]
    else:
        # All straight driving — use everything
        print("[WARNING] No turning data found! Model will only learn straight driving.")
        balanced = records

    np.random.shuffle(balanced)
    print(f"[Dataset] Balanced: {len(balanced)} records "
          f"({len(turning)} turning, {min(len(straight), len(turning)*2 if len(turning) > 0 else len(straight))} straight)")

    # Split
    train_recs, val_recs = train_test_split(balanced, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"[Dataset] Train: {len(train_recs)} | Val: {len(val_recs)}")

    train_ds = TraffiqDataset(train_recs, BATCH_SIZE, augment=True)
    val_ds   = TraffiqDataset(val_recs,   BATCH_SIZE, augment=False)

    # Build model
    model = build_dave2_model()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )

    # Callbacks
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_save = str(models_dir / "dave2_traffiq.h5")
    csv_log    = str(models_dir / "training_log.csv")

    cb_list = [
        callbacks.ModelCheckpoint(
            model_save, monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(csv_log),
    ]

    # Train
    print(f"\n[Training] Starting for up to {epochs} epochs...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
    )

    plot_training_curves(history)
    export_tflite(model)
    benchmark_inference(model)


# ─── PLOTTING ─────────────────────────────────────────────

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TRAFFIQ — DAVE-2 Training Results", fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss", color="#4A90D9")
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#E74C3C")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history.history["mae"],     label="Train MAE", color="#2ECC71")
    axes[1].plot(history.history["val_mae"], label="Val MAE",   color="#F39C12")
    axes[1].set_title("Mean Absolute Error (Steering)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(PROJECT_ROOT / "models" / "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] Training curves → {save_path}")


# ─── TFLITE EXPORT (for Raspberry Pi) ───────────────────

def export_tflite(model, train_ds):
    """Convert to TRUE INT8 quantized TFLite model"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32 # Or int8, depending on your Pi script

    # Provide a representative dataset to calibrate the quantization
    def representative_dataset():
        for i in range(10): # Grab 10 batches
            images, _ = train_ds[i]
            yield [images]

    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()


# ─── INFERENCE BENCHMARK ──────────────────────────────────

def benchmark_inference(model, n_runs: int = 100):
    """
    Simulate how fast the model will run on Raspberry Pi.
    Run on your PC to get a rough estimate (Pi will be ~3-5x slower).
    """
    import time
    dummy = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times[10:])  # skip warmup
    print(f"\n[Benchmark] Avg inference (PC): {avg:.1f} ms")
    print(f"[Benchmark] Est. on Pi 4B:      {avg * 3.5:.0f} ms  "
          f"({'✓ OK' if avg * 3.5 < 80 else '✗ Too slow — reduce model size'})")


# ─── ENTRY POINT ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAFFIQ DAVE-2 Trainer")
    parser.add_argument("--data_dir", type=str,
                        default="dataset/20260327_005825",
                        help="Path to dataset folder (relative to traffiq/)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs (default: 30)")
    args = parser.parse_args()

    train(args.data_dir, args.epochs)