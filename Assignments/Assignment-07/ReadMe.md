# Assignment 07: Transfer Learning & CNN Architectures on MNIST

This directory contains the implementation and comparative evaluation of standard Convolutional Neural Network (CNN) architectures and pre-trained deep learning models for handwritten digit classification.

---

## 📌 Overview
The goal of this assignment is to train and evaluate multiple deep learning architectures on the MNIST dataset to compare performance across custom and transfer learning setups.

### Architectures Evaluated
1. **AlexNet** (Custom Implementation)
2. **VGG16** (Pre-trained on ImageNet, Feature Extraction)
3. **ResNet50** (Pre-trained on ImageNet, Feature Extraction)
4. **EfficientNetB0** (Pre-trained on ImageNet, Feature Extraction)

---

## 🛠️ Data Preprocessing Pipeline
* **Dataset:** MNIST (subset of 2,000 training samples and 500 test samples for fast execution).
* **Channel Expansion:** Grayscale images converted from 1 channel to 3 channels (`28x28x1` → `28x28x3`).
* **Resizing:** Input images resized to `32x32x3` to meet the minimum input size requirements of pre-trained backbones.
* **Normalization:** Pixel values scaled to the range `[0, 1]`.
* **Encoding:** Labels one-hot encoded into 10 target classes.

---

## 📊 Experimental Results

All models were trained for **5 epochs** with the **Adam optimizer** (`lr=0.001`) and **Categorical Cross-Entropy loss**.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **VGG16** | **0.8120** | **0.8172** | **0.8136** | **0.8020** |
| **ResNet50** | 0.6980 | 0.7237 | 0.6819 | 0.6715 |
| **AlexNet** | 0.2200 | 0.0475 | 0.1860 | 0.0749 |
| **EfficientNetB0** | 0.1080 | 0.0108 | 0.1000 | 0.0195 |

---

## 🔑 Key Observations & Insights
* **Top Performer:** **VGG16** achieved the highest overall accuracy (**81.2%**) and F1-score, leveraging its fixed deep convolutional feature extractor effectively on smaller resized images.
* **Overfitting / Training Instability:** The custom **AlexNet** model showed rapid training convergence but suffered from high validation loss variance due to overfitting on the reduced training subset.
* **Input Resolution Sensitivity:** **EfficientNetB0** requires specific input normalization and higher native resolution scaling to activate pre-trained weights properly, leading to underperformance at `32x32`.

---

## 🚀 How to Run
1. Open [`DL_ASS_7.ipynb`](DL_ASS_7.ipynb) in Google Colab or your local Jupyter environment.
2. Ensure required dependencies are installed:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
