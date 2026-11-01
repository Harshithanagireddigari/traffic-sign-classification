"""
train.py
========
Train a Convolutional Neural Network (CNN) to classify traffic signs from the
German Traffic Sign Recognition Benchmark (GTSRB).

Pipeline
--------
1. Load labelled images from ``dataset/GTSRB`` (filenames come from the CSV
   files that ship with the dataset, read with pandas).
2. Preprocess every image: resize to 32x32 pixels and normalize pixel values
   from [0, 255] to [0, 1].
3. Build a CNN: Conv2D -> ReLU -> MaxPooling (x3) -> Flatten -> Dense -> Softmax.
4. Train with a validation split, early stopping and a model checkpoint.
5. Save the trained model to ``models/traffic_sign_cnn.keras`` so the app never
   has to retrain it.
6. Evaluate on the official test set and save accuracy/loss graphs, a confusion
   matrix and a classification report into ``models/``.

Run from inside the ``Traffic-Sign-Classification`` folder:

    python train.py
"""

import json
import os

import cv2  # OpenCV: image reading / resizing
import matplotlib

# Use a non-interactive backend so the script also works on headless machines
# and on Windows without opening plot windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # reads the GTSRB CSV files
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sign_classes import CLASS_NAMES

# ------------------------------- configuration --------------------------------
IMG_SIZE = 32                       # every image is resized to 32 x 32 pixels
NUM_CLASSES = len(CLASS_NAMES)      # 43 traffic sign classes
BATCH_SIZE = 64
EPOCHS = 30

DATA_DIR = os.path.join("dataset", "GTSRB")
TRAIN_DIR = os.path.join(DATA_DIR, "Final_Training", "Images")
TEST_DIR = os.path.join(DATA_DIR, "Final_Test", "Images")
TEST_CSV = os.path.join(TEST_DIR, "Test.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_sign_cnn.keras")
HISTORY_PNG = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_PNG = os.path.join(MODEL_DIR, "confusion_matrix.png")
METRICS_JSON = os.path.join(MODEL_DIR, "evaluation.json")


# ------------------------------- data loading ---------------------------------
def check_dataset():
    """Stop early with a helpful message if the GTSRB dataset is missing."""
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"GTSRB dataset not found at '{TRAIN_DIR}'.\n"
            "Download the German Traffic Sign Recognition Benchmark (GTSRB) and extract it so that:\n"
            f"  {TRAIN_DIR}/00000 ... 00042   (training images + one CSV per class)\n"
            f"  {TEST_CSV}\n"
            "See dataset/README.md for download links."
        )


def preprocess(img_bgr):
    """Resize a BGR image to 32x32 (RGB) - used for every image we load."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # OpenCV reads BGR, Keras likes RGB
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def load_training_data():
    """Load all training images using the per-class CSV files shipped with GTSRB."""
    images, labels = [], []
    for class_id, folder in enumerate(sorted(os.listdir(TRAIN_DIR))):
        csv_path = os.path.join(TRAIN_DIR, folder, f"GT-{folder}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path, sep=";")  # columns: Filename;Width;Height;...;ClassId
        for _, row in df.iterrows():
            img = cv2.imread(os.path.join(TRAIN_DIR, folder, row["Filename"]))
            if img is None:                  # gracefully skip unreadable files
                continue
            images.append(preprocess(img))
            labels.append(class_id)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_test_data():
    """Load the official GTSRB test set (labels come from Test.csv)."""
    df = pd.read_csv(TEST_CSV, sep=";")
    images, labels = [], []
    for _, row in df.iterrows():
        img = cv2.imread(os.path.join(TEST_DIR, row["Filename"]))
        if img is None:
            continue
        images.append(preprocess(img))
        labels.append(int(row["ClassId"]))
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# --------------------------------- the CNN ------------------------------------
def build_model():
    """
    CNN architecture:

    Conv2D(32) + ReLU  ->  MaxPooling      (learns edges / corners)
    Conv2D(64) + ReLU  ->  MaxPooling      (learns shapes)
    Conv2D(128)+ ReLU  ->  MaxPooling      (learns sign parts)
    Flatten            ->  turn the 4x4x128 feature maps into a vector
    Dense(256) + ReLU + Dropout            (classifier head)
    Dense(43) + Softmax                    (probability per traffic sign class)
    """
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),  # randomly disable neurons to prevent overfitting
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # labels are plain integers 0..42
        metrics=["accuracy"],
    )
    return model


# ------------------------------ evaluation plots ------------------------------
def save_history_plot(history, path):
    """Accuracy and loss curves for training vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="training")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="training")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Model loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, path):
    """Heatmap of true class vs predicted class on the test set."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Confusion matrix (GTSRB test set)")
    plt.xlabel("predicted class")
    plt.ylabel("true class")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return cm


# ----------------------------------- main -------------------------------------
def main():
    check_dataset()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[1/5] Loading GTSRB training images ...")
    X, y = load_training_data()
    X = X / 255.0  # normalize pixel values from [0, 255] to [0, 1]
    print(f"      {X.shape[0]} images loaded, per-image shape: {X.shape[1:]}")

    # stratified split keeps the class balance identical in train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[2/5] Building the CNN ...")
    model = build_model()
    model.summary()

    callbacks = [
        # stop when validation accuracy stops improving
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        # lower the learning rate when the loss plateaus
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        # save the best model seen during training
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    print("[3/5] Training ...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    model.save(MODEL_PATH)  # final save (best weights were restored by EarlyStopping)
    print(f"[4/5] Model saved to '{MODEL_PATH}'")

    print("[5/5] Evaluating on the official GTSRB test set ...")
    X_test, y_test = load_test_data()
    X_test = X_test / 255.0
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    save_confusion_matrix(y_test, y_pred, CONFUSION_PNG)
    save_history_plot(history, HISTORY_PNG)

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["accuracy"]),
        "num_train_images": int(X.shape[0]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "classification_report": report,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"      Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
    print(f"      Saved: {HISTORY_PNG}, {CONFUSION_PNG}, {METRICS_JSON}")


if __name__ == "__main__":
    main()
