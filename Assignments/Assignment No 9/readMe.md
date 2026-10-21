# Assignment 9: Vision Transformer (ViT) vs. CNN Image Classification

## Overview
This assignment explores image classification using a pre-trained Vision Transformer (ViT) model and compares its performance directly against a Convolutional Neural Network (CNN) baseline (ResNet18) using the Fashion-MNIST dataset.

## Implementation Details
- **Dataset:** Fashion-MNIST (subsampled to 5,000 training images and 1,000 test images for efficient experimentation).
- **Preprocessing:** Images converted to 3 channels and resized to 224x224 pixels to match the input requirements of pre-trained models. Normalized using standard ImageNet statistics.
- **Models Evaluated:**
  - **CNN Baseline:** `resnet18` (pretrained via `timm`)
  - **Vision Transformer:** `vit_tiny_patch16_224` (pretrained via `timm`)

## Performance Summary

| Model | Test Accuracy | Training Time (5 Epochs) | Inference Time |
| :--- | :--- | :--- | :--- |
| **CNN (ResNet18)** | 87.10% | 85.51s | 2.22s |
| **ViT (Tiny)** | 92.20% | 102.43s | 2.03s |

## Key Findings
- The Vision Transformer achieved a higher classification accuracy (**92.20%**) compared to the ResNet18 baseline (**87.10%**).
- While the CNN required slightly less training time, the ViT demonstrated competitive inference speeds.

## How to Run
You can run and inspect the complete code directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Darshan-12345/Deep-Learning-/blob/main/Assignments/Assignment-09)
