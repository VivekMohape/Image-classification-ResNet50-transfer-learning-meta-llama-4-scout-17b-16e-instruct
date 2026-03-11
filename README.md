# Image-classification-ResNet50-transfer-learning-meta-llama-4-scout-17b-16e-instruct

# Indian Clothing Image Classification (IndoFashion)

## Overview

This project implements an **image classification system for Indian ethnic clothing** using the IndoFashion dataset.

The goal is to train deep learning models capable of identifying different categories of Indian clothing such as sarees, kurtas, lehengas, sherwanis, and others.

Since the full dataset is large, a **balanced subset of 7500 images** was created by selecting **500 images per class across 15 categories**.

The project uses **transfer learning with pretrained convolutional neural networks** to achieve high classification accuracy.

---

# Dataset

Dataset source:

https://indofashion.github.io/

The original dataset contains **15 clothing categories**.

To meet the assignment requirements, a **subset dataset was created**.

| Classes | Images per class | Total Images |
| ------- | ---------------- | ------------ |
| 15      | 500              | 7500         |

During preprocessing:

* Corrupt images were removed
* Unsupported image formats were filtered
* Exactly **500 valid images per class** were selected

This ensures a **balanced dataset for training**.

---

# Data Preprocessing

Images were processed using PyTorch transforms.

The following preprocessing steps were applied:

* Resize images to **224 × 224**
* Random horizontal flipping
* Random rotation (10 degrees)
* Normalize using **ImageNet statistics**

Example transform pipeline:

```
Resize(224,224)
RandomHorizontalFlip
RandomRotation(10)
ToTensor
Normalize(ImageNet Mean & Std)
```

These augmentations improve model generalization.

---

# Train / Validation Split

The dataset was split into:

* **80% training data**
* **20% validation data**

Data loaders were used for efficient batch training.

Batch size:

```
32
```

---

# Model Architectures

Two transfer learning models were evaluated.

## ResNet50

ResNet introduces **residual connections**, allowing very deep networks to train effectively.

Key idea:

```
F(x) + x
```

This prevents vanishing gradients and enables better feature learning.

In this project:

* ImageNet pretrained weights were used
* The backbone network was frozen
* Only the final classification layer was trained

---

## EfficientNet-B0

EfficientNet uses **compound scaling**, balancing network depth, width, and resolution.

Advantages:

* fewer parameters
* efficient computation
* strong feature extraction

Similar to ResNet, the pretrained backbone was frozen and only the classifier was trained.

---

# Training Configuration

| Parameter      | Value                                 |
| -------------- | ------------------------------------- |
| Optimizer      | Adam                                  |
| Loss Function  | CrossEntropyLoss                      |
| Batch Size     | 32                                    |
| Epochs         | 15                                    |
| Learning Rate  | 0.001 (ResNet), 0.0005 (EfficientNet) |
| Scheduler      | StepLR                                |
| Early Stopping | Patience = 3                          |

A learning rate scheduler was used to gradually reduce the learning rate during training.

---

# Evaluation Metrics

Model performance was evaluated using:

* **Accuracy**
* **Confusion Matrix**
* **Training and validation loss curves**

---

# Results

| Model           | Best Validation Accuracy |
| --------------- | ------------------------ |
| ResNet50        | **85.33%**               |
| EfficientNet-B0 | **84.13%**               |

ResNet50 slightly outperformed EfficientNet on this dataset.

---

# Training Logs Example

Example output during training:

```
ResNet50 Epoch 7/15
Train Loss: 0.404
Val Loss: 0.436
Val Accuracy: 0.85
```

Training curves and confusion matrix visualizations were generated to analyze model performance.

---

# Potential Improvements

Several improvements could further increase model performance:

* Fine-tuning deeper layers of the pretrained networks
* Applying stronger data augmentation
* Using larger EfficientNet variants
* Model ensembling

These strategies may improve accuracy and robustness.

---

# Repository Structure

```
indo-fashion-image-classification
│
├── dataset_cleaning.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```


# Author

Vivek Mohape
