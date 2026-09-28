# Assignment 08: Sentiment Analysis on IMDB Reviews using Fine-Tuned BERT

This directory contains the implementation of fine-tuning a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model for binary sentiment classification using Hugging Face `transformers`, `datasets`, and `trainer` APIs.

---

## 📌 Overview
* **Dataset:** [IMDB Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) (Stanford NLP)
* **Task:** Binary Sentiment Classification (`0` = Negative, `1` = Positive)
* **Base Model:** `bert-base-uncased`
* **Framework:** PyTorch, Hugging Face Transformers, Datasets, Evaluate

---

## 🏗️ Workflow & Pipeline

1. **Dataset Preparation:**
   * Loaded the standard IMDB dataset split (`25,000` training, `25,000` testing).
   * Created a subset with `5,000` training samples and `1,000` test samples for efficient fine-tuning.

2. **Tokenization:**
   * Used `AutoTokenizer` from `bert-base-uncased`.
   * Applied truncation and padding to a maximum sequence length of **128 tokens**.

3. **Model Fine-Tuning:**
   * Initialized `AutoModelForSequenceClassification` with `num_labels = 2`.
   * Optimized using `Trainer` with:
     * **Learning Rate:** `2e-5`
     * **Batch Size:** 8 per device
     * **Weight Decay:** `0.01`
     * **Epochs:** 2

---

## 📊 Evaluation & Results

The fine-tuned model achieved strong classification accuracy on the test subset after 2 epochs:

| Metric | Epoch 1 | Epoch 2 (Final) |
| :--- | :---: | :---: |
| **Training Loss** | 0.1687 | **0.0769** |
| **Validation Loss** | 0.8265 | **0.8543** |
| **Accuracy** | 84.70% | **85.50%** |

---

## 🧪 Sample Predictions

The model accurately predicts sentiment labels for unseen custom input strings:

* **Input:** *"This movie was amazing and I really enjoyed it."*  
  $\rightarrow$ **Prediction:** `Positive`

* **Input:** *"The movie was boring and completely disappointing."*  
  $\rightarrow$ **Prediction:** `Negative`

---

## 🚀 How to Run
1. Open [`DL_Assignment_NO_8.ipynb`](DL_Assignment_NO_8.ipynb) in Google Colab.
2. Install the necessary dependencies:
   ```bash
   pip install transformers datasets evaluate accelerate torch
