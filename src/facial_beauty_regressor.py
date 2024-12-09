#!/usr/bin/env python
# coding: utf-8

from tqdm.notebook import tqdm
import pandas as pd
from retinaface import RetinaFace
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import cv2
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
from retinaface import RetinaFace

detector = RetinaFace(quality='normal')

BASE_PATH = './'
data = []
with open(f'{BASE_PATH}/labels.txt', 'r', encoding='utf-8') as labels_file:
    labels = labels_file.readlines()
    for label in tqdm(labels):
        row = label.rstrip('\n').split(' ')
        data.append(row)

df = pd.DataFrame(data, columns=['Filename', 'Beauty Rate'])

class BeautyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        filename, label = self.df.iloc[idx].values
        img_path = os.path.join(self.img_dir, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        faces = detector.predict(image)
        face = faces[0]
        try:
            image = cv2.resize(image[face['y1']:face['y2'], face['x1']:face['x2']], (256, 256))
        except Exception as e:
            image = cv2.resize(image, (256, 256))
            print(e)
        # image = cv2.resize(image[face['y1']:face['y2'], face['x1']:face['x2']], (256, 256))
        # image = transform(image)
        # image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(float(label), dtype=torch.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = BeautyDataset(df, f'{BASE_PATH}/Images/Images', transform=transform)

val_size = 0.2
indices = list(range(len(df)))
np.random.shuffle(indices)
split = int(np.floor(val_size * len(df)))
train_indices, val_indices = indices[split:], indices[:split]


train_ds = Subset(ds, train_indices)
val_ds = Subset(ds, val_indices)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


# model = torchvision.models.mobilenet_v3_small(num_classes=1)
model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.classifier.add_module("Linear", nn.Linear(1000, 1))
summary(model, (3, 256, 256))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
torch.save(model.state_dict(), './beauty_model_regressor.pth')

model.eval()

index = 0
images, labels = next(iter(val_loader))
image = images[index]
label = labels[index]

image = image.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
model = model.to(device)

with torch.no_grad():
    output = model(image)

predicted_value = output.item()

image = image.squeeze().cpu().numpy().transpose((1, 2, 0))



