import pandas as pd
import numpy as np

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Sequential, Conv2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from tqdm import tqdm

from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights

import csv
import os
import cv2
import math

from ImageUtils import resize_img, filter_img
from Config import IMG_SIZE, N_OF_IMGS, N_EPOCHS

train = pd.read_csv("./train.csv")
train.head()

imgs = []

# กำหนดจำนวนชุดข้อมูลทดสอบ เป็น %
valid_size_percentage = 0.1

print(f"\nLoading {N_OF_IMGS} images...\n")

for img_name in tqdm(train["id"][:N_OF_IMGS]):
    # path เต็มรูป
    img_path = os.path.join("./train", img_name.split("_")[0], img_name)

    # นำเข้ารูป
    img = imread(img_path, as_gray=True)

    filtered_img = filter_img(img)

    # plt.imshow(filtered_img, cmap="gray")
    # plt.show()

    resized_img = resize_img(filtered_img, IMG_SIZE)

    imgs.append(resized_img)

train_imgs = np.array(imgs)

train_labels = np.array(train["label"].values[:N_OF_IMGS])

train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(train_imgs, train_labels, test_size=valid_size_percentage)

train_dim = train_imgs.shape[0]
valid_dim = valid_imgs.shape[0]

# แปลง train imgs เป็น torch format
train_imgs = train_imgs.reshape(train_dim, 3, IMG_SIZE, IMG_SIZE)
train_imgs = torch.from_numpy(train_imgs).to(torch.float32)

# แปลง train labels เป็น torch format
train_labels = train_labels.astype(int)
train_labels = torch.from_numpy(train_labels).to(torch.long)

# แปลง valid imgs เป็น torch format
valid_imgs = valid_imgs.reshape(valid_dim, 3, IMG_SIZE, IMG_SIZE)
valid_imgs = torch.from_numpy(valid_imgs).to(torch.float32)

# แปลง valid labels เป็น torch format
valid_labels = valid_labels.astype(int)
valid_labels = torch.from_numpy(valid_labels).to(torch.long)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

optimizer = SGD(model.parameters(), lr=0.7, momentum=0.9)

criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# เก็บค่า training loss
train_losses = []

# เก็บค่า validation loss
valid_losses = []

# จำนวนรอบในการ train

print(f"\nStart training {N_EPOCHS} round...\n")

for epoch in tqdm(range(N_EPOCHS)):
    model.train()
    tr_loss = 0

    train_imgs_copy, train_labels_copy = Variable(train_imgs), Variable(train_labels)
    valid_imgs_copy, valid_labels_copy = Variable(valid_imgs), Variable(valid_labels)

    optimizer.zero_grad()

    output_train = model(train_imgs_copy)
    output_valid = model(valid_imgs_copy)

    train_labels_copy = train_labels_copy
    train_labels_copy = train_labels_copy
    valid_labels_copy = valid_labels_copy
    valid_labels_copy = valid_labels_copy

    loss_train = criterion(output_train, train_labels_copy)
    loss_valid = criterion(output_valid, valid_labels_copy)
    train_losses.append(loss_train.item())
    valid_losses.append(loss_valid.item())

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

model_dir = f"./models/model_{IMG_SIZE}x{IMG_SIZE}_{N_OF_IMGS}imgs_{N_EPOCHS}r.pt"

# torch.save(model.state_dict(), f"./models/mod`el_v{version}.pt")
print(f"\nCompleted training")
print(f"Save trained model to {model_dir}\n")

torch.save(model.state_dict(), model_dir)
