# Assignment 9: Image Classification using Vision Transformer (ViT) vs. CNN (ResNet18)

This repository contains an assignment implementation comparing the performance of a pre-trained **Convolutional Neural Network (ResNet18)** and a **Vision Transformer (ViT-Tiny)** on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## 📋 Overview

The objective of this assignment is to evaluate and contrast how traditional convolutional architectures and modern transformer-based architectures handle image classification tasks under identical training constraints.

* **Dataset:** Fashion-MNIST (Grayscale clothing images resized to $224 \times 224$ and converted to 3-channel tensors to match pre-trained model input requirements). Subsets used: **5,000 training samples** and **1,000 test samples**.
* **Models Compared:**
  1. **CNN:** `resnet18` (via `timm`)
  2. **Vision Transformer:** `vit_tiny_patch16_224` (via `timm`)
* **Hyperparameters:**
  * Epochs: `5`
  * Batch Size: `32`
  * Optimizer: `Adam` ($lr = 1 \times 10^{-4}$)
  * Learning Rate Scheduler: `StepLR` (step size = 3, gamma = 0.1)
  * Loss Function: `CrossEntropyLoss`

---

## 📊 Performance Summary

Both models were trained and evaluated on an NVIDIA GPU device (`cuda`).

| Model | Test Accuracy | Training Time | Inference Time |
| :--- | :---: | :---: | :---: |
| **CNN (ResNet18)** | 87.10% | 85.51s | 2.22s |
| **ViT (Tiny)** | **92.20%** | 102.43s | **2.03s** |

---

## 🛠️ Requirements & Dependencies

To run the notebook, make sure you have the following Python libraries installed:

```bash
pip install torch torchvision timm scikit-learn
