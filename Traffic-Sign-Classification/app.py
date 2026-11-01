"""
app.py
======
Streamlit web interface for the Traffic Sign Classification project.

Pages (sidebar):
    * Home               - project overview and ML pipeline explanation
    * Predict            - upload a JPG/JPEG/PNG image and classify it with the CNN
    * Model Performance  - accuracy/loss graphs, confusion matrix, report
    * About              - concepts, project structure, how to run

Run from inside the Traffic-Sign-Classification folder:

    streamlit run app.py

The trained model (models/traffic_sign_cnn.keras) is loaded once and cached,
so the app starts instantly without retraining.
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sign_classes import CLASS_NAMES
from predict import MODEL_PATH, preprocess_image

# ------------------------------- page setup -----------------------------------
st.set_page_config(
    page_title="Traffic Sign Classification - CNN",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small custom CSS to make the default Streamlit theme feel more professional
st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 1200px; }
        .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
                       padding: 1rem; text-align: center; }
        .pred-name  { font-size: 2rem; font-weight: 800; color: #0f766e; }
        .stProgress > div > div { background-color: #14b8a6 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- model loading (cached) ---------------------------
@st.cache_resource
def load_model():
    """Load the trained Keras model once; cached across reruns (st.cache_resource)."""
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH)


def model_available():
    return os.path.isfile(MODEL_PATH)


# ------------------------------- page: Home -----------------------------------
def page_home():
    st.title("🚦 Traffic Sign Classification")
    st.subheader("Convolutional Neural Network (CNN) · TensorFlow/Keras · GTSRB dataset")
    st.markdown(
        """
        This mini project trains a **Convolutional Neural Network (CNN)** on the
        **German Traffic Sign Recognition Benchmark (GTSRB)** — ~39,000 labelled
        training images across **43 traffic-sign classes** — and classifies new,
        user-uploaded images of traffic signs.

        **Pipeline:** upload image → resize to **32×32** → normalize pixels to **[0, 1]**
        → CNN (Conv2D + ReLU + MaxPooling → Flatten → Dense + Softmax) → predicted
        sign class + confidence.
        """
    )

    st.markdown("### 🧠 How the CNN works")
    cols = st.columns(4)
    concepts = [
        ("Conv2D + ReLU", "Slides small filters over the image to detect edges, shapes and patterns."),
        ("MaxPooling", "Keeps the strongest activations and shrinks the image (translation invariance)."),
        ("Flatten + Dense", "Turns the learned feature maps into a vector and classifies it."),
        ("Softmax", "Outputs a probability for each of the 43 sign classes (they sum to 100%)."),
    ]
    for col, (title, desc) in zip(cols, concepts):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("### 🧰 Technology stack")
    st.markdown(
        "- **Python** + **TensorFlow / Keras** (CNN model)\n"
        "- **OpenCV** (image reading & resizing) · **NumPy / Pandas** (arrays & CSV data)\n"
        "- **Matplotlib / Seaborn** (graphs) · **Streamlit** (this web UI)"
    )


# ------------------------------ page: Predict ---------------------------------
def page_predict():
    st.title("🔍 Predict a traffic sign")
    st.write("Upload a JPG, JPEG or PNG image of a traffic sign.")

    if not model_available():
        st.error(
            f"Trained model not found at `{MODEL_PATH}`.\n\n"
            "Train it first with `python train.py` (see README.md), then restart this app."
        )
        return

    uploaded = st.file_uploader(
        "Choose a traffic sign image", type=["jpg", "jpeg", "png"]
    )
    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return

    # PIL -> OpenCV (BGR) so we can reuse the exact preprocessing from predict.py
    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.markdown("**Uploaded image**")
        st.image(pil_img, use_container_width=True)
    with col_res:
        st.markdown("**Preprocessed (32×32, normalized)**")
        small = cv2.resize(img_bgr, (32, 32))
        st.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), width=160)
        st.caption("Resized to 32×32 px and scaled to [0, 1] before feeding the CNN.")

    if st.button("🚀 Predict", type="primary"):
        model = load_model()
        x = preprocess_image(img_bgr)                 # shape (1, 32, 32, 3)
        probs = model.predict(x, verbose=0)[0]         # 43 softmax probabilities
        top3 = np.argsort(probs)[::-1][:3]

        best = int(top3[0])
        st.markdown("---")
        st.markdown("### ✅ Prediction result")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f'<div class="pred-name">[{best:02d}] {CLASS_NAMES[best]}</div>',
                        unsafe_allow_html=True)
            st.metric("Confidence", f"{probs[best] * 100:.2f}%")
        with c2:
            st.markdown("**Top-3 predictions**")
            for rank, idx in enumerate(top3, start=1):
                idx = int(idx)
                st.write(f"{rank}. [{idx:02d}] {CLASS_NAMES[idx]}")
                st.progress(float(probs[idx]))


# ------------------------- page: Model Performance ----------------------------
def page_performance():
    st.title("📊 Model performance")
    metrics_path = os.path.join("models", "evaluation.json")
    history_png = os.path.join("models", "training_history.png")
    confusion_png = os.path.join("models", "confusion_matrix.png")

    if not os.path.isfile(metrics_path):
        st.warning(
            "No evaluation results yet. Run `python train.py` first — it saves "
            "`models/evaluation.json`, `training_history.png` and `confusion_matrix.png`."
        )
        return

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
    c2.metric("Test loss", f"{metrics['test_loss']:.4f}")
    c3.metric("Epochs trained", metrics["epochs_trained"])
    st.caption(f"Evaluated on the official GTSRB test set ({metrics['num_train_images']} training images were used).")

    if os.path.isfile(history_png):
        st.markdown("### Accuracy & loss curves")
        st.image(history_png, use_container_width=True)
    if os.path.isfile(confusion_png):
        st.markdown("### Confusion matrix (test set)")
        st.image(confusion_png, use_container_width=True)

    st.markdown("### Classification report")
    report = metrics["classification_report"]
    rows = []
    for name in CLASS_NAMES:
        if name in report:
            r = report[name]
            rows.append({
                "Class": name,
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1-score": round(r["f1-score"], 4),
                "Support": int(r["support"]),
            })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------- page: About ----------------------------------
def page_about():
    st.title("ℹ️ About this project")
    st.markdown(
        """
        **Traffic Sign Classification using Python and Machine Learning** — a B.Tech CSE
        (AI & ML) mini project.

        ### Project structure
        ```
        Traffic-Sign-Classification/
        ├── dataset/            # GTSRB dataset goes here (see dataset/README.md)
        ├── models/             # trained model + graphs are saved here
        ├── sign_classes.py     # the 43 human-readable class names
        ├── train.py            # trains and saves the CNN
        ├── predict.py          # classifies one image from the command line
        ├── app.py              # this Streamlit app
        ├── requirements.txt
        └── README.md
        ```

        ### ML concepts used
        - **Convolution (Conv2D):** small learnable filters slide over the image and
          produce feature maps (edges → shapes → sign parts).
        - **ReLU:** activation function `max(0, x)` — introduces non-linearity.
        - **MaxPooling:** 2×2 downsampling that keeps the strongest features.
        - **Flatten:** reshapes the final feature maps into a 1-D vector.
        - **Dense + Softmax:** classifier producing 43 probabilities.
        - **Dropout:** randomly disables neurons during training to reduce overfitting.
        - **Normalization:** pixel values scaled from [0, 255] to [0, 1] so the
          network trains faster and more stably.

        ### Run locally (Windows)
        ```bat
        python -m venv .venv
        .venv\\Scripts\\activate
        pip install -r requirements.txt
        python train.py            :: trains the CNN (needs dataset/GTSRB)
        python predict.py test.png :: classify one image
        streamlit run app.py       :: start the web UI
        ```
        """
    )


# --------------------------------- router -------------------------------------
def main():
    st.sidebar.title("🚦 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🔍 Predict", "📊 Model Performance", "ℹ️ About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "CNN · TensorFlow/Keras · OpenCV · GTSRB\nB.Tech CSE (AIML) mini project"
    )

    if page == "🏠 Home":
        page_home()
    elif page == "🔍 Predict":
        page_predict()
    elif page == "📊 Model Performance":
        page_performance()
    else:
        page_about()


if __name__ == "__main__":
    main()
