import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = ImageFolder("clean_subset_dataset",transform=transform)

train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size

train_dataset,val_dataset=random_split(dataset,[train_size,val_size])

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32)

def train_model(model,lr):

    model=model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    epochs=15

    for epoch in range(epochs):

        model.train()

        for images,labels in train_loader:

            images=images.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs,labels)

            loss.backward()
            optimizer.step()

        print("Epoch:",epoch)

    return model

resnet=torchvision.models.resnet50(weights="IMAGENET1K_V1")

for param in resnet.parameters():
    param.requires_grad=False

resnet.fc=nn.Linear(resnet.fc.in_features,15)

train_model(resnet,0.001)
