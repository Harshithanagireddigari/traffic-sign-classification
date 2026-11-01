"""
train.py
========
Complete Machine Learning Workflow for Traffic Sign Classification:
Step 1: Collect / Load the real GTSRB dataset (43 classes).
Step 2: Preprocess images (resize to 32x32, normalize [0, 1], encode labels 0..42).
Step 3: Split dataset: 70% Training, 15% Validation, 15% Testing.
Step 4: Train CNN (Conv2D -> ReLU -> MaxPool x3 -> Flatten -> Dense -> Softmax)
        and save to models/traffic_sign_cnn.keras.
Step 5: Evaluate model (Train/Val/Test accuracy, Precision, Recall, F1-score, Confusion Matrix).
Step 6: Test with unseen test images (Stop sign, No-entry, Speed-limit, Pedestrians, Road work).
Step 7: Deploy model ready for FastAPI / Render REST API.

Usage:
    python train.py
    python train.py --download  # automatically downloads dataset via kagglehub if missing
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sign_classes import CLASS_NAMES

# ------------------------------- Configuration --------------------------------
IMG_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)  # 43
BATCH_SIZE = 64
EPOCHS = 18

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "GTSRB")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_cnn.keras")
HISTORY_PNG = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_PNG = os.path.join(MODEL_DIR, "confusion_matrix.png")
METRICS_JSON = os.path.join(MODEL_DIR, "evaluation.json")


def ensure_dataset(download_if_missing=False):
    """Checks for GTSRB dataset or optionally downloads it via kagglehub."""
    if os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 1:
        return

    if download_if_missing:
        print("[Step 1] Dataset not found in dataset/GTSRB. Downloading via kagglehub...")
        try:
            import kagglehub
            download_path = kagglehub.dataset_download("meowmeowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
            print(f"         Dataset downloaded to {download_path}")
            os.makedirs(DATA_DIR, exist_ok=True)
            for item in os.listdir(download_path):
                src = os.path.join(download_path, item)
                dst = os.path.join(DATA_DIR, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            print(f"         Copied dataset into '{DATA_DIR}' successfully.")
            return
        except Exception as e:
            print(f"Automatic download failed: {e}")

    raise FileNotFoundError(
        f"GTSRB dataset not found inside '{DATA_DIR}'.\n"
        "Please download the dataset from Kaggle or INI Benchmark and extract it to dataset/GTSRB/.\n"
        "Or run with: python train.py --download"
    )


def preprocess_image(img_bgr):
    """Step 2: Resize to 32x32 RGB and convert."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))


def load_all_images():
    """Step 2: Load images from either official or Kaggle folder structure."""
    images = []
    labels = []

    # 1. Try Kaggle Train folder layout (Train/0 .. Train/42)
    kaggle_train = os.path.join(DATA_DIR, "Train")
    if not os.path.isdir(kaggle_train):
        kaggle_train = os.path.join(DATA_DIR, "train")

    # 2. Try official benchmark layout (Final_Training/Images/00000 .. 00042)
    official_train = os.path.join(DATA_DIR, "Final_Training", "Images")

    target_dir = None
    if os.path.isdir(kaggle_train):
        target_dir = kaggle_train
    elif os.path.isdir(official_train):
        target_dir = official_train
    elif os.path.isdir(DATA_DIR):
        # Check if direct numbered subfolders exist in DATA_DIR
        subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if any(d.isdigit() for d in subdirs):
            target_dir = DATA_DIR

    if not target_dir:
        raise FileNotFoundError(f"Could not locate training image folders in '{DATA_DIR}'.")

    print(f"         Scanning image folders in '{target_dir}'...")
    folders = sorted([f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))])

    for folder in folders:
        try:
            class_id = int(folder)
        except ValueError:
            continue
        if class_id >= NUM_CLASSES:
            continue

        folder_path = os.path.join(target_dir, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".ppm", ".jpg", ".jpeg")):
                fpath = os.path.join(folder_path, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                images.append(preprocess_image(img))
                labels.append(class_id)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def build_cnn():
    """Step 4: CNN Architecture: 3 Conv blocks -> Flatten -> Dense -> Softmax."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Train Traffic Sign CNN on GTSRB dataset.")
    parser.add_argument("--download", action="store_true", help="Download GTSRB dataset automatically via kagglehub")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("TRAFFIC SIGN CLASSIFICATION - COMPLETE ML WORKFLOW")
    print("=" * 70)

    # ------------------ Step 1: Collect / Load the real GTSRB dataset ------------------
    print("\n[Step 1] Collecting / checking GTSRB dataset...")
    ensure_dataset(download_if_missing=args.download)

    # ------------------ Step 2: Preprocess images ------------------
    print("\n[Step 2] Preprocessing images:")
    print("         * Resizing to 32 x 32 RGB")
    print("         * Converting to NumPy arrays")
    print("         * Normalizing pixel values to [0, 1]")
    print("         * Encoding labels 0 to 42")
    X, y = load_all_images()
    X = X / 255.0  # Normalize to [0, 1]
    print(f"         Total preprocessed dataset: {X.shape[0]} images | Shape: {X.shape[1:]}")

    # ------------------ Step 3: Split the dataset (70% Train, 15% Val, 15% Test) ------------------
    print("\n[Step 3] Splitting dataset into three stratified sets:")
    print("         * Training:   70% (Model learning)")
    print("         * Validation: 15% (Hyperparameter tuning & early stopping)")
    print("         * Testing:    15% (Unseen holdout evaluation)")

    # Stratified split: 85% train+val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # From the 85%, take 70/85 (~82.35%) for train, and 15/85 (~17.65%) for validation
    val_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=42, stratify=y_train_val
    )

    print(f"         -> Training set:   {X_train.shape[0]:5d} images ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"         -> Validation set: {X_val.shape[0]:5d} images ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"         -> Testing set:    {X_test.shape[0]:5d} images ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

    # ------------------ Step 4: Train the CNN ------------------
    print("\n[Step 4] Training the CNN:")
    print("         * Architecture: Conv2D(32) -> Conv2D(64) -> Conv2D(128) -> Dense(256) -> Dense(43)")
    print("         * Optimizer: Adam(0.001) | Loss: sparse_categorical_crossentropy")
    model = build_cnn()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_PATH)
    print(f"         [OK] Model successfully saved to '{MODEL_PATH}'")

    # ------------------ Step 5: Evaluate the model ------------------
    print("\n[Step 5] Evaluating model performance across sets:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="weighted", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro", zero_division=0
    )

    print("-" * 55)
    print(f"  Training Accuracy:   {train_acc * 100:6.2f}% | Loss: {train_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc * 100:6.2f}% | Loss: {val_loss:.4f}")
    print(f"  Testing Accuracy:    {test_acc * 100:6.2f}% | Loss: {test_loss:.4f}")
    print(f"  Weighted Precision:  {precision * 100:6.2f}%")
    print(f"  Weighted Recall:     {recall * 100:6.2f}%")
    print(f"  Weighted F1-score:   {f1 * 100:6.2f}%")
    print(f"  Macro F1-score:      {macro_f1 * 100:6.2f}%")
    print("-" * 55)

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix (Unseen GTSRB Test Set)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=150)
    plt.close()

    # Save Accuracy/Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training Acc", color="#2dd4bf")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc", color="#34d399")
    axes[0].set_title("Model Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training Loss", color="#fbbf24")
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#fb7185")
    axes[1].set_title("Model Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(HISTORY_PNG, dpi=150)
    plt.close(fig)

    report_dict = classification_report(
        y_test, y_test_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    eval_data = {
        "status": "ready",
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_f1": float(macro_f1),
        "num_total_images": int(X.shape[0]),
        "num_train_images": int(X_train.shape[0]),
        "num_val_images": int(X_val.shape[0]),
        "num_test_images": int(X_test.shape[0]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "classification_report": report_dict,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    print(f"         [OK] Evaluation report saved to: {METRICS_JSON}")
    print(f"         [OK] Confusion matrix saved to:  {CONFUSION_PNG}")
    print(f"         [OK] Training history saved to:  {HISTORY_PNG}")

    # ------------------ Step 6: Test with new/unseen images ------------------
    print("\n[Step 6] Testing with unseen test images:")
    sample_test_classes = [
        (14, "Stop sign"),
        (17, "No-entry sign"),
        (2, "Speed-limit sign (50 km/h)"),
        (27, "Pedestrians sign"),
        (25, "Road work sign"),
    ]

    for class_id, label in sample_test_classes:
        indices = np.where(y_test == class_id)[0]
        if len(indices) > 0:
            sample_idx = indices[0]
            test_img = np.expand_dims(X_test[sample_idx], axis=0)
            preds = model.predict(test_img, verbose=0)[0]
            pred_id = int(np.argmax(preds))
            conf = float(preds[pred_id]) * 100
            correct = "[CORRECT]" if pred_id == class_id else "[INCORRECT]"
            print(f"         * Expected: {label:<28} | Predicted: [{pred_id:02d}] {CLASS_NAMES[pred_id]:<36} ({conf:5.1f}%) -> {correct}")

    # ------------------ Step 7: Ready for deployment ------------------
    print("\n[Step 7] Deployment readiness:")
    print(f"         * Trained weights ready at: '{MODEL_PATH}'")
    print(f"         * FastAPI REST API ready in: 'Traffic-Sign-Classification/server.py'")
    print(f"         * Launch API: uvicorn server:app --host 0.0.0.0 --port 8000")
    print("=" * 70)


if __name__ == "__main__":
    main()
