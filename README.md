# 🚦 Traffic Sign Classification Using CNN

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CNN-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-yellow)

A Machine Learning mini project that identifies and classifies traffic signs using Convolutional Neural Networks (CNN).

The project demonstrates image preprocessing, feature extraction, model training, classification and prediction through an interactive user interface.

## 📌 Project Overview

Traffic signs play an important role in road safety and traffic management. Recognizing these signs automatically can help intelligent transportation systems and autonomous vehicles understand road conditions.

This project explores how Deep Learning can be used to classify traffic signs from images.

Users can upload a traffic sign image and obtain a predicted category with its confidence score.

## 🎯 Objectives

* Develop a traffic sign classification system using CNN.
* Understand image preprocessing and feature extraction.
* Apply supervised machine learning techniques.
* Train and evaluate a classification model.
* Build an interactive interface for image prediction.
* Explore practical applications of AI in transportation.

## 🛠️ Technologies Used

| Technology         | Purpose                         |
| ------------------ | ------------------------------- |
| Python             | ML implementation               |
| TensorFlow / Keras | Deep Learning                   |
| CNN                | Image classification            |
| OpenCV             | Image processing                |
| NumPy              | Numerical operations            |
| Pandas             | Dataset handling                |
| Matplotlib         | Visualization                   |
| React              | Frontend development            |
| Vite               | Development server              |
| TensorFlow.js      | Browser-based CNN demonstration |
| Streamlit          | Python application interface    |

## 🧠 Machine Learning Concepts

The project demonstrates the following concepts:

1. **Supervised Learning:** Learning from labelled traffic sign images.
2. **Convolutional Neural Network:** Extracting visual features from images.
3. **Convolution Layer:** Detecting edges, shapes and patterns.
4. **ReLU Activation:** Introducing non-linearity.
5. **Max Pooling:** Reducing feature map dimensions.
6. **Flattening:** Converting feature maps into a one-dimensional representation.
7. **Softmax:** Producing class probabilities.
8. **Dropout:** Reducing overfitting.
9. **Model Evaluation:** Measuring classification performance.

## ⚙️ Project Structure

```text
traffic-sign-classification/
│
├── Traffic-Sign-Classification/
│   ├── dataset/
│   │   └── README.md
│   │
│   ├── models/
│   │   └── README.md
│   │
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   ├── sign_classes.py
│   ├── requirements.txt
│   └── README.md
│
├── src/
│   ├── components/
│   ├── lib/
│   ├── utils/
│   ├── App.tsx
│   ├── index.css
│   └── main.tsx
│
├── index.html
├── package.json
├── package-lock.json
├── vite.config.ts
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Harshithanagireddigari/traffic-sign-classification.git
```

### 2. Navigate to the Project

```bash
cd traffic-sign-classification
```

### 3. Install Frontend Dependencies

```bash
npm install
```

### 4. Start the Frontend

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

## 🐍 Running the Python Implementation

Navigate to the Python project:

```bash
cd Traffic-Sign-Classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Train the CNN Model

```bash
python train.py
```

### Run the Prediction Application

```bash
streamlit run app.py
```

Follow the dataset instructions in the `dataset/README.md` file before training.

## 📊 Model Architecture

The CNN architecture uses the following sequence:

```text
Input Image (32 × 32 × 3)
          |
          v
   Convolution Layer
          |
          v
     ReLU Activation
          |
          v
     Max Pooling
          |
          v
   Convolution Layer
          |
          v
     ReLU Activation
          |
          v
     Max Pooling
          |
          v
       Flatten
          |
          v
    Fully Connected
          |
          v
    Softmax Output
          |
          v
 Predicted Traffic Sign
```

## 📷 Application Screenshots

Screenshots of the application will be added here.

| Section            | Screenshot  |
| ------------------ | ----------- |
| Homepage           | Coming soon |
| Live Demo          | Coming soon |
| Dashboard          | Coming soon |
| ML Concepts        | Coming soon |
| Prediction Results | Coming soon |

## 📈 Expected Results

The application is designed to provide:

* Predicted traffic sign category.
* Classification confidence.
* Training and validation accuracy.
* Training and validation loss.
* Model performance visualizations.

Actual evaluation results should be recorded after training and testing the model.

## 🔮 Future Enhancements

* Real-time traffic sign recognition using a webcam.
* Integration with autonomous driving systems.
* Recognition of Indian traffic signs.
* Mobile application integration.
* Voice-based traffic sign announcements.
* Improved handling of unrelated images.

## 🎓 Academic Information

**Project Title:** Traffic Sign Classification Using CNN

**Project Type:** Mini Project

**Department:** Computer Science and Engineering

**Specialization:** Artificial Intelligence and Machine Learning

## 👩‍💻 Author

**Harshitha Nagireddigari**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: [Harshithanagireddigari](https://github.com/Harshithanagireddigari)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning and Deep Learning.**
