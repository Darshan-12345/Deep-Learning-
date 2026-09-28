# Assignment 01: Image Classification on CIFAR-10 using Artificial Neural Networks (ANN)

This directory contains the implementation of a Multi-Layer Perceptron (MLP) / Feedforward Neural Network built with TensorFlow and Keras to classify images from the **CIFAR-10** dataset.

---

## 📌 Overview
The goal of this assignment is to build, train, and evaluate a basic Deep Learning sequential model to perform multi-class image classification across 10 object categories.

### Key Highlights
* **Dataset:** CIFAR-10 (50,000 training images, 10,000 testing images)
* **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
* **Framework:** TensorFlow / Keras

---

## 🏗️ Model Architecture

The network uses a simple Feedforward / Dense architecture:

1. **Input / Flatten Layer:** Reshapes input images from `(32, 32, 3)` into a 1D vector of `3,072` features.
2. **Hidden Layer 1:** Dense layer with `256` neurons and **ReLU** activation.
3. **Hidden Layer 2:** Dense layer with `128` neurons and **ReLU** activation.
4. **Output Layer:** Dense layer with `10` neurons and **Softmax** activation for multi-class probability output.

* **Total Parameters:** ~820,874 trainable parameters

---

## ⚙️ Data Preprocessing & Training Setup
* **Normalization:** Pixel intensities scaled from `[0, 255]` to `[0.0, 1.0]` by dividing by `255.0`.
* **Optimizer:** Adam
* **Loss Function:** `sparse_categorical_crossentropy`
* **Evaluation Metric:** Accuracy
* **Training Epochs:** 5

---

## 📊 Performance & Results

* **Test Accuracy:** **~44.83%**
* **Model Artifact:** Saved as `cifar10_model.keras` after training.

---

## 🚀 How to Run
1. Open [`Assignment_NO_1.ipynb`](https://github.com/Darshan-12345/Deep-Learning-/blob/main/Assignments/Assignment-01/Assignment_NO_1.ipynb) in Google Colab or your local Jupyter Notebook environment.
2. Install necessary libraries if not already available:
   ```bash
   pip install tensorflow matplotlib numpy
