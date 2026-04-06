"""
============================================================
 TRAFFIQ — Competition AI Training Script (Round 1)

 What this does:
   1. Accounts for higher resolution initial inputs (e.g., 640x480).
   2. Changes the model output to be an array of size 2: [Speed, Direction],
      both within the range [-1, 1], as per the Round 1 specifications.
   3. Upgrades the basic DAVE-2 architecture to use a more robust feature
      extractor (like MobileNet-v2 base or an enhanced custom CNN) to better
      recognize objects, track lines, and handle lighting variations.
   4. Exports to TFLite (INT8) for final deployment on the Raspberry Pi.

 Usage:
   python3 training/train_competition_model.py --data_dir dataset/xyz
============================================================
"""

import os
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2

# ─── CONFIGURATION ────────────────────────────────────────
# Using larger inputs to maintain detail for object/lane detection
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
IMG_CHANNELS = 3
BATCH_SIZE   = 32
EPOCHS       = 40
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH  = "models/competition_model.h5"
TFLITE_SAVE_PATH = "models/competition_model.tflite"
# ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── DATA AUGMENTATION ────────────────────────────────────

def augment_brightness(image: np.ndarray) -> np.ndarray:
    """Randomly brighten/darken to simulate variable lighting"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = 0.4 + np.random.uniform()   # 0.4 → 1.4
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def augment_shadow(image: np.ndarray) -> np.ndarray:
    """Add a random shadow stripe"""
    h, w = image.shape[:2]
    x1, x2 = np.random.randint(0, w, 2)
    shadow_mask = np.zeros_like(image[:, :, 0])
    pts = np.array([[x1, 0], [x2, h], [w, h], [w, 0]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [pts], 1)
    image = image.copy().astype(np.float32)
    image[shadow_mask == 1] *= 0.5
    return np.clip(image, 0, 255).astype(np.uint8)

def augment_flip(image: np.ndarray, steering: float, speed: float):
    """Mirror the image horizontally and negate steering. Speed stays same."""
    return cv2.flip(image, 1), -steering, speed

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    In this updated preprocess, we ensure the image retains enough 
    resolution and context for object detection & lanes.
    """
    # Assuming initial image could be 640x480
    # Crop the top portion (e.g. background/sky) and bottom (car hood)
    # The actual crop values will depend on your exact camera angle
    height, width = image.shape[:2]
    crop_top = int(height * 0.3)
    crop_bottom = int(height * 0.95)
    
    image = image[crop_top:crop_bottom, :, :]
    
    # Resize to the model's required input size (224x224 for MobileNet base)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Keeping RGB instead of YUV since pre-trained feature extractors expect RGB
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
        images, targets = [], []

        for rec in batch:
            img_path = str(PROJECT_ROOT / rec["image_path"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Read steering and speed/throttle from JSON
            # Competition asks for: array containing two values [Speed, Direction]
            direction = float(rec.get("steering", 0.0))
            # Fallback to 0.5 if throttle/speed isn't explicitly recorded
            speed = float(rec.get("throttle", 0.5)) 

            if self.augment:
                img = augment_brightness(img)
                if np.random.random() > 0.5:
                    img = augment_shadow(img)
                if np.random.random() > 0.5:
                    img, direction, speed = augment_flip(img, direction, speed)

            img = preprocess(img)
            # MobileNet pre-processing is usually [-1, 1] scale or custom. 
            # We scale [0, 1] here, adjust as needed.
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            # Targets scale expected to be [-1, 1]
            targets.append([speed, direction])

        if len(images) == 0:
            return (np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32),
                    np.zeros((1, 2), dtype=np.float32))

        return np.array(images), np.array(targets)

    def on_epoch_end(self):
        np.random.shuffle(self.records)


# ─── ARCHITECTURE WITH OBJECT DETECTION CAPACITY ──────────

def build_competition_model():
    """
    Updated Architecture:
    Uses a robust base (MobileNetV2) to better extract complex features
    such as obstacles and variable lighting, followed by regression heads
    to output [Speed, Direction].
    """
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # We use MobileNetV2 (without classification head) as our feature extractor.
    # It is lightweight enough for Raspberry Pi but much stronger than basic DAVE-2.
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base base_model initially (optional, but good for small datasets)
    base_model.trainable = False 
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    # Decision Making Layers
    x = layers.Dense(128, activation='elu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='elu')(x)
    
    # OUTPUT: 2 values [Speed, Direction] activated by tanh to keep range [-1, 1]
    outputs = layers.Dense(2, activation='tanh', name='speed_and_direction')(x)
    
    model = models.Model(inputs, outputs, name="Competition_Net")
    return model


# ─── TRAINING PIPELINE ────────────────────────────────────

def train(data_dir: str, epochs: int):
    data_path = PROJECT_ROOT / data_dir
    label_file = data_path / "labels.json"

    if not label_file.exists():
        raise FileNotFoundError(f"labels.json not found in {data_path}")

    with open(label_file) as f:
        records = json.load(f)

    print(f"\n[Dataset] Loaded {len(records)} records from {data_path}")

    # For multi-output (Speed+Direction), basic data balancing gets tricky.
    # We'll just shuffle for now, but consider balancing based on obstacles later.
    np.random.shuffle(records)
    
    train_recs, val_recs = train_test_split(records, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"[Dataset] Train: {len(train_recs)} | Val: {len(val_recs)}")

    train_ds = TraffiqDataset(train_recs, BATCH_SIZE, augment=True)
    val_ds   = TraffiqDataset(val_recs,   BATCH_SIZE, augment=False)

    model = build_competition_model()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",        # MSE measures loss on BOTH speed and direction outputs together
        metrics=["mae"]
    )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_save = str(models_dir / "competition_model.h5")
    csv_log    = str(models_dir / "training_log_comp.csv")

    cb_list = [
        callbacks.ModelCheckpoint(
            model_save, monitor="val_loss", save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(csv_log),
    ]

    print(f"\n[Training] Starting for up to {epochs} epochs...\n")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb_list,
    )

    plot_training_curves(history)
    export_tflite(model, val_ds) # Pass representative dataset
    benchmark_inference(model)

# ─── UTILITIES (Plotting & TFLite) ────────────────────────
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Competition Model Training Results", fontsize=14, fontweight='bold')
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("MSE Loss (Speed + Direction)")
    axes[0].legend()
    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].legend()
    plt.savefig(str(PROJECT_ROOT / "models" / "comp_training_curves.png"))

def export_tflite(model, dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Force full integer quantization for optimal Pi Performance
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    def representative_dataset():
        for i in range(min(5, len(dataset))):
            images, _ = dataset[i]
            yield [images]

    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    
    save_path = str(PROJECT_ROOT / TFLITE_SAVE_PATH)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"[Export] Saved TFLite file: {save_path}")

def benchmark_inference(model, n_runs: int = 50):
    import time
    dummy = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    avg = np.mean(times[10:])
    print(f"\n[Benchmark] Avg inference (PC): {avg:.1f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Competition Model Trainer")
    parser.add_argument("--data_dir", type=str, default="dataset/your_dataset_folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.data_dir, args.epochs)
