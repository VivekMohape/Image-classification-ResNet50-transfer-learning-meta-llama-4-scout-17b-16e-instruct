Clothing Image Classification using Transfer Learning and LLM

🔗 Live Demo
https://img-classifier-efficientnet-llama4-scout.streamlit.app/

This project implements an image classification system for Indian ethnic clothing using the IndoFashion dataset.

The objective is to build a machine learning pipeline capable of automatically classifying clothing items such as sarees, kurtas, lehengas, sherwanis, and other traditional Indian garments.

Because the original dataset is large, a balanced subset dataset was created by selecting 500 valid images per class, resulting in 7500 images across 15 clothing categories.

The models were implemented using PyTorch and trained using transfer learning with pretrained convolutional neural networks.

Project Approach

The overall approach follows a typical deep learning pipeline for computer vision tasks:

Dataset Cleaning and Preparation

Balanced Subset Dataset Creation

Image Preprocessing and Data Augmentation

Model Training using Transfer Learning

Model Evaluation and Performance Analysis

Model Comparison

Interactive Deployment using Streamlit

Two deep learning architectures were evaluated:

ResNet50

EfficientNet-B0

The repository also includes an experimental LLM-based visual classifier using Llama Vision for comparison with deep learning approaches.

Dataset

Dataset Source:

https://indofashion.github.io/

The IndoFashion dataset contains images of various Indian ethnic clothing items collected from e-commerce platforms.

For this project, a smaller balanced dataset was created to ensure manageable training time.

Classes	Images per Class	Total Images
15	500	7500

Clothing categories include items such as:

Saree

Lehenga

Kurta

Sherwani

Mojari

Palazzo

Dupatta

Blouse

Leggings

Nehru Jacket

Dataset Cleaning

Before training the models, the dataset was cleaned to ensure data quality.

Cleaning steps included:

Removing corrupt images

Filtering unsupported image formats

Validating images using the PIL image verification

Randomly selecting 500 valid images per class

This process ensures:

Balanced dataset

Stable model training

Reduced bias between classes

Data Preprocessing

Images were preprocessed using PyTorch transforms.

Preprocessing pipeline:

Resize images to 224 × 224

Random horizontal flipping

Random rotation (±10°)

Convert images to tensors

Normalize using ImageNet mean and standard deviation

Normalization values:

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

These preprocessing steps help the model learn more robust visual representations.

Train / Validation Split

The dataset was split into training and validation sets.

Split	Percentage
Training	80%
Validation	20%

PyTorch DataLoader objects were used to efficiently load images in batches during training.

Batch size used:

32
Model Architecture
ResNet50

ResNet50 is a deep convolutional neural network architecture that introduces residual connections to solve the vanishing gradient problem.

Residual learning allows the model to learn transformations of the form:

F(x) + x

This enables training of deeper networks without degradation.

In this project:

ImageNet pretrained weights were used

Backbone layers were frozen

Only the final classification layer was retrained for the 15 clothing categories

EfficientNet-B0

EfficientNet is a modern CNN architecture that scales:

network depth

network width

input resolution

using compound scaling.

Advantages:

Strong feature extraction

High accuracy with fewer parameters

Efficient computation

Similar to ResNet50:

pretrained ImageNet weights were used

backbone layers were frozen

the classifier layer was replaced with a 15-class output layer

Training Process

Both models were trained using transfer learning.

Training configuration:

Parameter	Value
Optimizer	Adam
Loss Function	CrossEntropyLoss
Batch Size	32
Epochs	15
Learning Rate	0.001 (ResNet)
Learning Rate	0.0005 (EfficientNet)
Scheduler	StepLR
Early Stopping	Patience = 3
Learning Rate Scheduler

The StepLR scheduler gradually reduces the learning rate after fixed intervals to stabilize training.

Early Stopping

Training stops automatically if validation loss does not improve for several epochs, preventing overfitting.

Training Performance

The training curves illustrate the model learning behavior across epochs.

Observations:

Both models improve rapidly in early epochs

Training stabilizes after approximately 10–12 epochs

Validation loss plateaus indicating model convergence

No major overfitting observed

Evaluation Metrics

Model performance was evaluated using:

Validation Accuracy

Confusion Matrix

Training and Validation Loss Curves

Confusion Matrix

The confusion matrix visualizes how often each class is correctly predicted.

Key Observations

High prediction accuracy for visually distinct categories:

BLOUSE

LEGGINGS

PALAZZO

SHERWANIS

LEHENGA

Some minor confusion exists between visually similar classes:

Kurta Mens vs Women Kurta

Women Mojari vs Mens Mojari

Most predictions fall along the diagonal, indicating correct classification.

Final Results
Model	Validation Accuracy
ResNet50	~86%
EfficientNet-B0	~85%

ResNet50 achieved slightly higher validation accuracy on the dataset.

Streamlit Demo

A live interactive demo is available.

🔗 Live App
https://img-classifier-efficientnet-llama4-scout.streamlit.app/

The application allows users to upload clothing images and obtain predictions from the trained model.

Features include:

Upload clothing images

Display Top-3 predicted classes

Show prediction confidence

Visualize prediction probabilities

Model selection (CNN vs LLM classifier)

Repository Structure
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
Future Improvements

Possible improvements to increase model performance:

Fine-tuning deeper layers of pretrained models

Using larger EfficientNet architectures

Applying stronger data augmentation

Training with larger dataset subsets

Ensemble learning across multiple architectures
