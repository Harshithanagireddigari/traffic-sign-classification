# 🚦 Traffic Sign Classification Using Python and Machine Learning

A B.Tech CSE (Artificial Intelligence & Machine Learning) mini project that
classifies **traffic signs from uploaded images** using a **Convolutional
Neural Network (CNN)** trained on the **German Traffic Sign Recognition
Benchmark (GTSRB)**.

![stack](https://img.shields.io/badge/Python-3.10+-blue) ![tf](https://img.shields.io/badge/TensorFlow-Keras-orange) ![streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![gtsrb](https://img.shields.io/badge/Dataset-GTSRB-green)

---

## 1. Objective

Build a machine-learning application that identifies and classifies traffic
signs from user-uploaded images (JPG / JPEG / PNG) using a CNN, and presents
the prediction together with a confidence score and a full model-evaluation
dashboard.

## 2. Technology stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML / Deep Learning | TensorFlow + Keras, CNN (Conv2D, MaxPooling, Flatten, Dense, ReLU, Softmax, Dropout) |
| Image processing | OpenCV (`cv2`) |
| Data handling | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Evaluation | scikit-learn (confusion matrix, classification report) |
| Frontend | Streamlit |
| Dataset | GTSRB — German Traffic Sign Recognition Benchmark (43 classes, ~39k train / ~12.8k test images) |

## 3. Project structure

```
Traffic-Sign-Classification/
├── dataset/
│   └── README.md                 # where to put GTSRB (download links inside)
├── models/
│   └── README.md                 # trained model + graphs are saved here
├── sign_classes.py               # the 43 human-readable class names
├── train.py                      # trains the CNN and saves it to models/
├── predict.py                    # classify a single image from the CLI
├── app.py                        # Streamlit web application
├── requirements.txt
└── README.md                     # this file
```

## 4. Dataset (GTSRB)

Download the German Traffic Sign Recognition Benchmark and extract it into
`dataset/GTSRB/` (see `dataset/README.md` for the expected layout):

- Official: https://benchmark.ini.rub.de/gtsrb_dataset.html
- Kaggle mirror: https://www.kaggle.com/datasets/meowmeowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## 5. Installation & running (Windows)

```bat
:: 1. create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

:: 2. install dependencies
pip install -r requirements.txt

:: 3. train the CNN (requires dataset/GTSRB) - run ONCE
python train.py

:: 4a. classify a single image from the command line
python predict.py path\to\sign.png

:: 4b. start the Streamlit web app
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## 6. What the app does

### A. Image upload
Upload any **JPG / JPEG / PNG** traffic-sign image; the app displays it.

### B. Preprocessing
Every image is resized to **32 × 32 pixels**, converted to RGB, normalized to
pixel range **[0, 1]** and reshaped into a NumPy array of shape
`(1, 32, 32, 3)` — exactly the shape the CNN was trained on.

### C. The CNN model (`train.py`)
```
Input (32, 32, 3)
Conv2D(32, 3x3, ReLU)  -> MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU)  -> MaxPooling2D(2x2)
Conv2D(128,3x3, ReLU)  -> MaxPooling2D(2x2)
Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(43, Softmax)
```
Trained with the Adam optimizer, `sparse_categorical_crossentropy` loss,
early stopping and a model checkpoint. The best model is saved to
`models/traffic_sign_cnn.keras` and reloaded by the app — **no retraining on
every start**.

### D. Prediction
For an uploaded image the app shows:

1. the **predicted traffic-sign category** (class id),
2. the **traffic-sign name** (e.g. "Stop", "Speed limit (50 km/h)"),
3. the **confidence in percent**,
4. the **top-3 predictions** with progress bars.

### E. Model evaluation
`train.py` evaluates the model on the official GTSRB test set and saves:

- **training vs validation accuracy** graph,
- **training vs validation loss** graph,
- **confusion matrix** heatmap,
- **classification report** (precision / recall / F1 per class) as JSON.

The Streamlit *Model Performance* page displays all of them (real measured
values — nothing is hardcoded).

## 7. Error handling

- Missing dataset → `train.py` prints download instructions and exits.
- Missing trained model → `app.py` / `predict.py` show a clear message telling
  you to run `python train.py` first.
- Invalid / unreadable image → `predict.py` exits with a helpful message; the
  Streamlit uploader only accepts JPG/JPEG/PNG.

## 8. Sample output

```
Top-3 predictions for 'stop_sign.png':
  1. [class 14] Stop                                              confidence:  99.87%
  2. [class 13] Yield                                             confidence:   0.09%
  3. [class 17] No entry                                          confidence:   0.02%
```

## 9. ML concepts covered

Convolution · ReLU · Max-pooling · Flatten · Dense layers · Softmax ·
Dropout · Normalization · One-hot/integer labels · Train/validation/test split ·
Overfitting & early stopping · Confusion matrix · Precision / Recall / F1.
