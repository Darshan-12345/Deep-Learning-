# Assignment 05: Sequence Classification & Performance Analysis (SimpleRNN vs. LSTM vs. GRU)

This directory contains the implementation and comparative evaluation of standard Recurrent Neural Network architectures—**SimpleRNN**, **LSTM (Long Short-Term Memory)**, and **GRU (Gated Recurrent Unit)**—for sentiment classification on the IMDB movie reviews dataset.

---

## 📌 Overview
* **Dataset:** [IMDB Movie Reviews](https://keras.io/api/datasets/imdb/) (25,000 training reviews, 25,000 testing reviews)
* **Task:** Binary Sentiment Classification (`0` = Negative, `1` = Positive)
* **Vocabulary Size:** Top 10,000 words (`max_features = 10000`)
* **Sequence Length:** Padded/Truncated to 200 words (`maxlen = 200`)
* **Framework:** TensorFlow / Keras, Scikit-Learn, Pandas

---

## 🏗️ Model Architecture Pipeline

Each sequence model shares an identical input and output architecture for a fair performance comparison:

1. **Embedding Layer:** Maps integer-encoded vocabulary tokens to 128-dimensional dense vectors (`output_dim = 128`).
2. **Recurrent Layer:** Evaluates one of three standard recurrent units with 64 hidden units:
   * **SimpleRNN(64)**
   * **LSTM(64)**
   * **GRU(64)**
3. **Output Layer:** Single Dense neuron with **Sigmoid** activation for binary output prediction.

---

## ⚙️ Training Setup
* **Optimizer:** Adam
* **Loss Function:** `binary_crossentropy`
* **Metrics:** Accuracy
* **Batch Size:** 128
* **Epochs:** 3

---

## 📊 Experimental Results & Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | **86.81%** | 87.31% | **86.14%** | **86.72%** |
| **GRU** | 86.17% | **88.47%** | 83.18% | 85.75% |
| **SimpleRNN** | 82.37% | 81.03% | 84.53% | 82.74% |

---

## 🔑 Key Observations & Insights
* **Gated Architectures Excel:** Both **LSTM** and **GRU** significantly outperform the basic **SimpleRNN**, demonstrating superior handling of long-term sequential dependencies in text data.
* **Top Overall Performer:** **LSTM** achieved the highest overall accuracy (**86.81%**) and balanced F1-Score (**86.72%**).
* **High Precision:** **GRU** yielded the highest precision (**88.47%**), making fewer false-positive positive predictions compared to the other architectures.

---

## 🚀 How to Run
1. Open [`DL_Ass_NO_5.ipynb`](DL_Ass_NO_5.ipynb) in Google Colab.
2. Ensure the required dependencies are installed:
   ```bash
   pip install tensorflow pandas numpy scikit-learn
