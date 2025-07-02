# Compact Convolutional Transformers (CCT) on CIFAR-10

## 📌 Project Overview

This project implements and explores **Compact Convolutional Transformers (CCT)** for **image classification on the CIFAR-10 dataset**, leveraging a hybrid approach combining **CNNs for local feature extraction** with **Transformers for global relationship modeling**. This approach enables effective learning on small image datasets with limited computational resources.

---

## 🚀 Features

-   CCT model architecture implemented in TensorFlow/Keras.
-   Custom data augmentation pipeline for improving generalization.
-   Training and evaluation on CIFAR-10 and CIFAR-100.
-   Exploration of variants:

    -   Different patch sizes and strides.
    -   Optimizer comparison (AdamW vs. SGD with momentum).
    -   Deeper Transformer layers with advanced augmentation.
    -   Performance testing on CIFAR-100 to evaluate scalability.

-   Clear, commented, and modular code suitable for learning and extension.

---

## 📂 Repository Structure

```
/src/
    train_cct.py         # Main training script for CCT on CIFAR datasets
    evaluate_cct.py      # Evaluation script on test data
    model_cct.py         # CCT model architecture
    utils.py             # Utility functions for data loading and preprocessing
requirements.txt         # Python dependencies
README.md                # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/compact-cct-cifar10.git
cd compact-cct-cifar10
pip install -r requirements.txt
```

---

## 🏃 Usage

Train the CCT model:

```bash
python train_cct.py
```

Evaluate the model on the test set:

```bash
python evaluate_cct.py
```

You can modify hyperparameters directly in `train_cct.py` to explore different variants (e.g., optimizer, patch size, number of transformer layers).

---

## 📊 Results

| Variant                         | Accuracy (Validation) | Notes                                      |
| ------------------------------- | --------------------- | ------------------------------------------ |
| Baseline CCT CIFAR-10           | \~80%                 | Good generalization                        |
| CCT with SGD                    | \~80%                 | Stable convergence, lower memory footprint |
| CCT deeper + heavy augmentation | \~55%                 | Demonstrates generalization challenges     |
| CCT on CIFAR-100                | \~40%                 | Higher complexity dataset                  |

Visual results with accuracy/loss curves can be added here for demonstration.

---

## 🛠️ Technologies Used

-   **TensorFlow / Keras** for model implementation and training.
-   **NumPy, Matplotlib** for data handling and visualization.
-   **Python 3.9+** for scripting.

---

## 📚 Concepts Explored

-   **Compact Convolutional Transformers (CCT)**
-   Vision Transformers
-   CNN-based tokenization
-   Positional Encoding
-   Stochastic Depth Regularization
-   Data Augmentation for image datasets
-   Optimizer tuning (AdamW vs. SGD)
