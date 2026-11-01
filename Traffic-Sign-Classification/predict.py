"""
predict.py
==========
Classify a single traffic-sign image with the trained CNN.

Usage (from inside the Traffic-Sign-Classification folder):

    python predict.py path/to/sign.png
    python predict.py images/stop_sign.jpg

The script loads the model saved by train.py, preprocesses the image
(resize to 32x32, normalize to [0, 1]) and prints the top-3 predictions
with their confidence in percent.
"""

import os
import sys

import cv2
import numpy as np

from sign_classes import CLASS_NAMES

IMG_SIZE = 32
MODEL_PATH = os.path.join("models", "traffic_sign_cnn.keras")


def load_model_safe():
    """Load the saved Keras model, with friendly errors if anything is missing."""
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        sys.exit("TensorFlow is not installed. Run:  pip install -r requirements.txt")
    if not os.path.isfile(MODEL_PATH):
        sys.exit(
            f"Trained model not found at '{MODEL_PATH}'.\n"
            "Train it first with:  python train.py"
        )
    return load_model(MODEL_PATH)


def preprocess_image(img_bgr):
    """
    Preprocess one OpenCV (BGR) image exactly like the training data:
    RGB conversion -> resize to 32x32 -> normalize to [0, 1] -> add batch dim.
    Returns a NumPy array of shape (1, 32, 32, 3).
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    arr = img.astype(np.float32) / 255.0  # normalize pixel values
    return np.expand_dims(arr, axis=0)     # (1, 32, 32, 3)


def predict_image(model, image_path, top_k=3):
    """Return the top-k predictions as (class_id, name, confidence) tuples."""
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(
            f"Could not read '{image_path}'. Please provide a valid JPG, JPEG or PNG image."
        )
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]        # softmax output: 43 probabilities
    top_indices = np.argsort(probs)[::-1][:top_k]  # highest probabilities first
    return [(int(i), CLASS_NAMES[int(i)], float(probs[i])) for i in top_indices]


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <path_to_image.jpg|jpeg|png>")
    image_path = sys.argv[1]
    model = load_model_safe()
    results = predict_image(model, image_path, top_k=3)

    print(f"\nTop-3 predictions for '{image_path}':")
    for rank, (class_id, name, conf) in enumerate(results, start=1):
        print(f"  {rank}. [class {class_id:02d}] {name:<48} confidence: {conf * 100:6.2f}%")


if __name__ == "__main__":
    main()
