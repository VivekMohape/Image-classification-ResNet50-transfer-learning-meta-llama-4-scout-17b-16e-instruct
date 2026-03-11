# Clothing Image Classification using transfer learning and LLM



This project implements an **image classification system for Indian ethnic clothing** using the **IndoFashion dataset**.

The objective is to train deep learning models that can automatically classify clothing items such as sarees, kurtas, lehengas, sherwanis, and other traditional Indian garments.

Since the original dataset is large, a **balanced subset dataset was created by selecting 500 valid images per class**, resulting in **7500 images across 15 categories**.

The models were implemented using **PyTorch** and trained using **transfer learning with pretrained convolutional neural networks**.

## Approach

The approach followed in this project includes the following steps:

1. Dataset cleaning and preprocessing to remove corrupt images.
2. Creation of a balanced subset dataset with 500 images per class.
3. Application of image preprocessing and augmentation techniques.
4. Training deep learning models using transfer learning.
5. Evaluating model performance using validation accuracy and confusion matrix.
6. Comparing multiple architectures to determine the best performing model.


Two architectures were evaluated:

* ResNet50
* EfficientNet-B0

---

# Dataset

Dataset source:

https://indofashion.github.io/

The dataset contains **15 clothing categories**.

For this project, a subset dataset was created to make training manageable.

| Classes | Images per Class | Total Images |
| ------- | ---------------- | ------------ |
| 15      | 500              | 7500         |

During preprocessing:

* Corrupt images were removed
* Unsupported formats were filtered
* Exactly **500 valid images per class** were selected

This ensures a **balanced dataset for training**.

---

# Data Preprocessing

Images were processed using PyTorch transforms.

Preprocessing steps:

* Resize images to **224 × 224**
* Random horizontal flipping
* Random rotation (10°)
* Convert to tensor
* Normalize using **ImageNet mean and standard deviation**

These augmentations help improve model generalization.

---

# Train / Validation Split

The dataset was split into:

* **80% training data**
* **20% validation data**

DataLoader objects were used for efficient mini-batch training.

Batch size:

```
32
```

---

# Model Architectures

## ResNet50

ResNet introduces **residual connections**, allowing deep neural networks to train effectively.

Key concept:

Residual connections help mitigate the **vanishing gradient problem**, enabling deeper architectures.

In this project:

* ImageNet pretrained weights were used
* Backbone layers were frozen
* Only the **final classification layer** was trained

---

## EfficientNet-B0

EfficientNet scales network depth, width, and resolution simultaneously using compound scaling.

Advantages:

* Efficient architecture
* Strong feature extraction
* Fewer parameters compared to traditional CNNs

Similarly, pretrained weights were used and only the classifier layer was trained.

---

# Training Configuration

| Parameter      | Value                                  |
| -------------- | -------------------------------------- |
| Optimizer      | Adam                                   |
| Loss Function  | CrossEntropyLoss                       |
| Batch Size     | 32                                     |
| Epochs         | 15                                     |
| Learning Rate  | 0.001 (ResNet) / 0.0005 (EfficientNet) |
| Scheduler      | StepLR                                 |
| Early Stopping | Patience = 3                           |

A learning rate scheduler gradually reduced the learning rate during training.

Early stopping prevented unnecessary training once validation loss stopped improving.

---

# Training Performance

## Validation Loss and Accuracy

![Training Curves](results/training_curves.png)

Observations:

* Both models show rapid improvement during early epochs.
* ResNet50 achieves slightly **lower validation loss**.
* Accuracy stabilizes after approximately **10–12 epochs**, indicating convergence.

---

# Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

The confusion matrix shows how well the model distinguishes between clothing categories.

Each row represents the **true class**, while each column represents the **predicted class**.

### Key Observations

**High Accuracy for Distinct Categories**

The model performs well on clearly distinguishable clothing items such as:

* BLOUSE
* LEGGINGS
* PALAZZO
* SHERWANIS
* LEHENGA

**Confusion Between Similar Clothing**

Some classes show minor misclassification due to visual similarity:

* Kurta Mens vs Women Kurta
* Women Mojari vs Mens Mojari

Most predictions appear along the **diagonal of the confusion matrix**, indicating correct classifications.

---

# Final Results

| Model           | Validation Accuracy |
| --------------- | ------------------- |
| ResNet50        | **~86%**            |
| EfficientNet-B0 | **~85%**            |

ResNet50 achieved slightly higher validation accuracy on this dataset.

---

# Streamlit Demo

This repository includes a **Streamlit application** that allows users to upload an image and obtain predictions from the trained model.

Features:

* Upload clothing image
* Display **Top-3 predicted classes**
* Show prediction confidence
* Visualize prediction probabilities



# Repository Structure

```
indo-fashion-image-classification
│
├── train.py
├── dataset_cleaning.py
├── utils.py
├── app.py
├── requirements.txt
├── README.md
│
├── models
│   └── best_EfficientNet.pth
│
├── results
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── model_results.csv
```

---

# Potential Improvements

Future improvements could further increase model performance:

* Fine-tuning deeper layers of pretrained networks
* Applying stronger data augmentation
* Using larger EfficientNet variants
* Ensemble learning with multiple models


