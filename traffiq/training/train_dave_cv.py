"""
============================================================
 TRAFFIQ v2 — Updated Training Script
 File: training/train_v2.py

 KEY CHANGES FROM train_dave2.py:
   1. Output is now [Speed, Direction] — not just steering
   2. Input handles 640×480 (preprocessed down to 66×200)
   3. Two separate output heads — one per value
   4. Loss weights: direction penalized more than speed
   5. Labels now require BOTH speed and direction recorded
   6. Augmentation includes lighting color shifts
      (to handle arena lighting variation)

 Usage:
   python3 training/train_v2.py --data_dir ./dataset/SESSION
   python3 training/train_v2.py --data_dir ./dataset/SESSION --epochs 100
============================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # ADDED: Headless backend to prevent QT XCB crashes
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Add parent folder to path so we can import cv_pipeline
sys.path.append(str(Path(__file__).parent.parent))
from scripts.cv_pipeline import run_pipeline, normalize_lighting, crop_frame, preprocess_for_cnn

# ─── CONFIGURATION ────────────────────────────────────────
IMG_HEIGHT        = 66
IMG_WIDTH         = 200
IMG_CHANNELS      = 3
BATCH_SIZE        = 64
EPOCHS            = 100          # Increased — let EarlyStopping decide
LEARNING_RATE     = 1e-4
VALIDATION_SPLIT  = 0.2
MODEL_SAVE_PATH   = "models/traffiq1_v2.h5"
TFLITE_SAVE_PATH  = "models/traffiq1_v2.tflite"

# Loss weights — direction mistakes crash the car,
# speed mistakes just slow it down. Weight accordingly.
SPEED_LOSS_WEIGHT     = 0.4
DIRECTION_LOSS_WEIGHT = 0.6
# ──────────────────────────────────────────────────────────


# ─── AUGMENTATION ─────────────────────────────────────────
# Same as v1 PLUS a new lighting color shift augmentation
# specifically for the arena's variable lighting condition.

def augment_brightness(image: np.ndarray) -> np.ndarray:
    """Randomly brighten or darken the image (same as v1)."""
    hsv    = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = 0.4 + np.random.uniform()
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_lighting_color(image: np.ndarray) -> np.ndarray:
    """
    Simulates arena lighting color variation.

    The arena may have warm (yellow) or cool (blue) lighting.
    This augmentation shifts the overall color temperature of
    the image randomly — warm or cool — to teach the model
    to ignore lighting color when making decisions.

    HOW: Shift the hue and saturation channels in HSV
    randomly within a small range. This changes the overall
    color cast of the image without affecting brightness.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Shift hue by a small random amount (±15 degrees out of 180)
    hue_shift = np.random.uniform(-15, 15)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

    # Slightly vary saturation (±20%)
    sat_factor = 0.8 + np.random.uniform() * 0.4
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_shadow(image: np.ndarray) -> np.ndarray:
    """Add a random shadow stripe (same as v1)."""
    h, w = image.shape[:2]
    x1, x2 = np.random.randint(0, w, 2)
    shadow_mask = np.zeros_like(image[:, :, 0])
    pts = np.array([[x1, 0], [x2, h], [w, h], [w, 0]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [pts], 1)
    image = image.copy().astype(np.float32)
    image[shadow_mask == 1] *= 0.5
    return np.clip(image, 0, 255).astype(np.uint8)


def augment_flip(image: np.ndarray, direction: float):
    """Mirror image + negate direction (same as v1). Speed unchanged."""
    return cv2.flip(image, 1), -direction


# ─── DATASET LOADER ───────────────────────────────────────

class TraffiqDatasetV2(tf.keras.utils.Sequence):
    """
    Loads (image, [speed, direction]) pairs.

    Key difference from v1:
      - Each label is a 2-element array [speed, direction]
        not a single float.
      - Raw images are 640×480 (or as collected).
        The cv_pipeline preprocesses them down to 66×200 YUV.
    """
    def __init__(self, records: list, batch_size: int, augment: bool = False):
        self.records    = records
        self.batch_size = batch_size
        self.augment    = augment

    def __len__(self):
        return len(self.records) // self.batch_size

    def __getitem__(self, idx):
        batch = self.records[idx * self.batch_size : (idx + 1) * self.batch_size]
        images, labels = [], []

        for rec in batch:
            # img = cv2.imread(rec["image_path"])
            # ADDED: Correctly scope the dataset image path from the data_dir via absolute mapping created in train
            actual_path = rec.get("image_path_abs", rec["image_path"])
            img = cv2.imread(actual_path)
            
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # direction = float(rec["direction"])
            # speed     = float(rec["speed"])
            direction = float(rec["steering"])
            speed     = float(rec["throttle"])

            if self.augment:
                # Always apply lighting augmentations
                img = augment_brightness(img)
                img = augment_lighting_color(img)

                # 50% chance each for shadow and flip
                if np.random.random() > 0.5:
                    img = augment_shadow(img)
                if np.random.random() > 0.5:
                    img, direction = augment_flip(img, direction)
                    # Speed is symmetric — stays the same when flipped

            # Use cv_pipeline preprocessing:
            # normalize → crop → resize → YUV
            normalized = normalize_lighting(img)
            cropped    = crop_frame(normalized)
            processed  = preprocess_for_cnn(cropped)   # returns float32/255

            images.append(processed)
            labels.append([speed, direction])

        return np.array(images), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        np.random.shuffle(self.records)


# ─── MODEL ARCHITECTURE ───────────────────────────────────

def build_traffiq_v2_model():
    """
    Modified DAVE-2 with two output heads.

    SHARED BACKBONE:
      Same 5 Conv2D layers as DAVE-2.
      Shared because lane-following features
      (edges, curves) are useful for BOTH speed and
      direction decisions.

    TWO OUTPUT HEADS:
      Direction head → predicts steering [-1, 1]
      Speed head     → predicts throttle [-1, 1]

    WHY SEPARATE HEADS, NOT Dense(2)?
      Direction and speed have different learning
      dynamics. Direction changes every frame.
      Speed changes slowly and conservatively.
      Separate heads allow independent gradient flow
      — the optimizer can tune each output without
      one interfering with the other.

    WHY tanh ON BOTH?
      Output range must be [-1, 1] as per spec.
      tanh naturally constrains to this range.
    """
    inputs = tf.keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name='image_input'
    )

    # ── Convolutional backbone ────────────────────────────
    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu',
                      name='conv1')(inputs)
    x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu',
                      name='conv2')(x)
    x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu',
                      name='conv3')(x)
    x = layers.Conv2D(64, (3, 3), activation='elu', name='conv4')(x)
    x = layers.Conv2D(64, (3, 3), activation='elu', name='conv5')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.Flatten(name='flatten')(x)

    # ── Shared dense layers ───────────────────────────────
    x = layers.Dense(100, activation='elu', name='dense1')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    x = layers.Dense(50, activation='elu', name='dense2')(x)

    # ── Direction head ────────────────────────────────────
    dir_x     = layers.Dense(10, activation='elu', name='dir_dense')(x)
    direction = layers.Dense(1, activation='tanh', name='direction')(dir_x)

    # ── Speed head ────────────────────────────────────────
    spd_x = layers.Dense(10, activation='elu', name='spd_dense')(x)
    speed = layers.Dense(1, activation='tanh', name='speed')(spd_x)

    # ── Concatenate into [speed, direction] output ───────
    # Order matches the spec: output = [Speed, Direction]
    output = layers.Concatenate(name='output')([speed, direction])

    model = tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name='TRAFFIQ_v2'
    )
    return model


# ─── TRAINING ─────────────────────────────────────────────

def train(data_dir: str, epochs: int):

    # ── Load labels ──────────────────────────────────────
    label_file = Path(data_dir) / "labels.json"
    if not label_file.exists():
        raise FileNotFoundError(f"labels.json not found in {data_dir}")

    with open(label_file) as f:
        records = json.load(f)

    # ADDED: Safely resolve absolute path because running from parent directory crashes relative paths
    for r in records:
        r["image_path_abs"] = str(Path(data_dir) / "images" / os.path.basename(r.get("image_path", "")))

    # Check that labels have both speed and direction
    sample = records[0]
    # if "direction" not in sample or "speed" not in sample:
    if "steering" not in sample or "throttle" not in sample:
        raise KeyError(
            "labels.json must contain 'direction' and 'speed' fields.\n"
            "If you collected data with the old collect_data.py (which\n"
            "used 'steering' instead of 'direction'), rename the field:\n"
            "  python3 utils/migrate_labels.py --data_dir " + data_dir
        )

    print(f"\n[Dataset] Loaded {len(records)} records from {data_dir}")

    # ── Balance steering distribution ────────────────────
    # straight = [r for r in records if abs(r["direction"]) < 0.05]
    # turning  = [r for r in records if abs(r["direction"]) >= 0.05]
    straight = [r for r in records if abs(r["steering"]) < 0.05]
    turning  = [r for r in records if abs(r["steering"]) >= 0.05]
    balanced = turning + straight[:len(turning)]
    np.random.shuffle(balanced)
    print(f"[Dataset] Balanced: {len(balanced)} records "
          f"({len(turning)} turning, {len(straight[:len(turning)])} straight)")

    # ── Split ────────────────────────────────────────────
    train_recs, val_recs = train_test_split(
        balanced, test_size=VALIDATION_SPLIT, random_state=42
    )
    train_ds = TraffiqDatasetV2(train_recs, BATCH_SIZE, augment=True)
    val_ds   = TraffiqDatasetV2(val_recs,   BATCH_SIZE, augment=False)

    # ── Build and compile model ───────────────────────────
    model = build_traffiq_v2_model()
    model.summary()

    # Custom loss: weighted MSE
    # Direction errors weighted 60%, speed errors 40%
    # because direction is more critical for survival
    def weighted_mse(y_true, y_pred):
        speed_loss     = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
        direction_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
        return (SPEED_LOSS_WEIGHT * speed_loss +
                DIRECTION_LOSS_WEIGHT * direction_loss)

    def speed_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

    def direction_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=weighted_mse,
        metrics=[speed_mae, direction_mae]
    )

    # ── Callbacks ────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    cb_list = [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger("models/training_log_v2.csv"),
    ]

    # ── Train ────────────────────────────────────────────
    print(f"\n[Training] Starting for up to {epochs} epochs...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
    )

    plot_training_curves(history)
    export_tflite(model, data_dir)
    benchmark_inference(model)

    return model


# ─── PLOTTING ─────────────────────────────────────────────

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "TRAFFIQ v2 — Training Results [Speed, Direction]",
        fontsize=14, fontweight='bold'
    )

    # Total loss
    axes[0].plot(history.history["loss"],
                 label="Train Loss", color="#4A90D9")
    axes[0].plot(history.history["val_loss"],
                 label="Val Loss",   color="#E74C3C")
    axes[0].set_title("Weighted Loss (total)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speed MAE
    axes[1].plot(history.history["speed_mae"],
                 label="Train",  color="#2ECC71")
    axes[1].plot(history.history["val_speed_mae"],
                 label="Val",    color="#27AE60")
    axes[1].set_title("Speed MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.08, color='red', linestyle='--',
                    linewidth=1, label='Target')

    # Direction MAE
    axes[2].plot(history.history["direction_mae"],
                 label="Train", color="#F39C12")
    axes[2].plot(history.history["val_direction_mae"],
                 label="Val",   color="#E67E22")
    axes[2].set_title("Direction MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0.05, color='red', linestyle='--',
                    linewidth=1, label='Target')

    plt.tight_layout()
    plt.savefig("models/training_curves_v2.png", dpi=150)
    print("[Saved] Training curves → models/training_curves_v2.png")
    # plt.show() # Commented out to prevent Qt display crash


# ─── TFLITE EXPORT ────────────────────────────────────────

def export_tflite(model, data_dir: str):
    """
    Full INT8 quantization with calibration dataset.
    This is the version that actually runs at <80ms on Pi.
    (Dynamic-range-only quantization gave us 111ms before.)
    """
    import random

    with open(Path(data_dir) / "labels.json") as f:
        records = json.load(f)
    sample = random.sample(records, min(300, len(records)))

    def representative_dataset():
        for rec in sample:
            # img = cv2.imread(rec["image_path"])
            img_path_abs = str(Path(data_dir) / "images" / os.path.basename(rec.get("image_path", "")))
            img = cv2.imread(img_path_abs)
            if img is None:
                continue
            img        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            normalized = normalize_lighting(img)
            cropped    = crop_frame(normalized)
            processed  = preprocess_for_cnn(cropped)
            yield [processed[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_SAVE_PATH) / 1024
    print(f"\n[TFLite INT8] Saved → {TFLITE_SAVE_PATH}  ({size_kb:.1f} KB)")


# ─── INFERENCE BENCHMARK ──────────────────────────────────

def benchmark_inference(model, n_runs: int = 100):
    import time
    dummy = np.random.rand(
        1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
    ).astype(np.float32)
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times[10:])
    print(f"\n[Benchmark] Avg inference (PC): {avg:.1f} ms")
    est_pi = avg * 3.5
    status = '✓ OK' if est_pi < 80 else '✗ Too slow'
    print(f"[Benchmark] Est. on Pi 4B:       {est_pi:.0f} ms  ({status})")


# ─── ENTRY POINT ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAFFIQ v2 Trainer")
    parser.add_argument("--data_dir", type=str, default="/home/remon/Documents/Trafiic_car_autonomous/traffiq/dataset/20260327_005825")
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    args = parser.parse_args()

    train(args.data_dir, args.epochs)