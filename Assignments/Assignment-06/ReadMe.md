# Assignment 06: Sugarcane Leaf Disease Caption Generation (CNN + LSTM)

This directory contains the implementation of an end-to-end Image Captioning model using a hybrid **CNN + LSTM** neural network architecture. The model extracts visual feature representations from sugarcane leaf images using transfer learning and generates descriptive text captions indicating leaf health and specific disease conditions.

---

## 📌 Overview
* **Dataset:** [Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/sunilgautam/sugarcane-leaf-disease-dataset) (2,521 total images across 5 classes).
* **Classes:** Yellow, Rust, Mosaic, Healthy, RedRot.
* **Task:** Automatically generate text descriptions (e.g., *"startseq a sugarcane leaf affected by red rot endseq"*) for leaf images.
* **Framework:** TensorFlow / Keras, Scikit-Learn, Pandas.

---

## 🏗️ System Architecture

1. **Feature Extractor (CNN Backbone):**
   * Pre-trained **MobileNetV2** (weights pre-trained on ImageNet).
   * Dense feature representation of size `(1280,)` extracted per image after removing classification heads.

2. **Sequence Model (Text Input):**
   * Text Tokenizer with `vocab_size = 15`.
   * **Embedding Layer:** Maps text sequence tokens to 256-dimensional vectors.
   * **LSTM Layer:** Processes token sequences with 256 hidden units.

3. **Decoder / Merger Network:**
   * Combines visual features (`Dense(256)`) and text sequence features (`LSTM(256)`) via element-wise addition (`Add()`).
   * Output Dense layer with **Softmax** activation predicts the next word probability over the vocabulary size.

---

## ⚙️ Preprocessing & Training Details
* **Train/Test Split:** 80% training (2,016 images) and 20% testing (505 images), stratified by disease class.
* **Sequence Padding:** Post-padded token sequences to `max_length = 9`.
* **Optimizer:** Adam
* **Loss Function:** `sparse_categorical_crossentropy`
* **Batch Size:** 64
* **Epochs:** 10

---

## 📊 Model Performance

| Metric | Training | Validation |
| :--- | :---: | :---: |
| **Accuracy (Epoch 10)** | **99.07%** | **97.63%** |
| **Loss (Epoch 10)** | 0.0266 | 0.0806 |

---

## 🚀 How to Run
1. Open [`DL_06.ipynb`](DL_06.ipynb) in Google Colab.
2. Upload your `kaggle.json` API key to download the dataset automatically.
3. Run all cells sequentially to extract MobileNetV2 features, train the combined CNN-LSTM model, and perform auto-regressive caption inference on test leaf images.
