import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from tqdm import tqdm
import matplotlib.pyplot as plt

from helpers.transform import transform

# กำหนด batch
batch_size = 32
# กำหนดจำนวน class (ผลไม้ 10 ชนิด)
num_classes = 10
# จำนวนรอบในการ train
num_epochs = 25
# folder ที่ทำการบันทึก model
model_save_path = f"../models/ResNet_v3_{num_epochs}epc.pt"

dataset_path = "../datasets/train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลดข้อมูลรูปทั้งหมด
dataset = ImageFolder(root=dataset_path, transform=transform)

# แยกเป็นชุด train และ valid
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size])

# สร้างตัวโหลดข้อมูลพร้อมกำหนด batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# สร้าง model พร้อมกำหนด weight
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# แก้ไข model layer เป็น 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# นำ model เข้า device
model = model.to(device)

# กำหนด loss function และ optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(f"\nLoaded \033[94m{len(dataset)}\033[0m dataset to model")

print(f"\nStart training \033[91mResNet\033[0m model \033[91m{num_epochs}\033[0m epochs\n")

print("\033[91m", end="")
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # เริ่มทดสอบ
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        print("\n")
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loss /= len(train_dataset)
            valid_loss /= len(valid_dataset)
            accuracy = 100.0 * correct / total
            print(f"\033[0mEpoch {epoch + 1}/{num_epochs} - "f"Train Loss: {train_loss:.4f}, "f"Valid Loss: {valid_loss:.4f}, "f"Valid Accuracy: {accuracy:.2f}%\033[91m")
        print()
print("\033[0m", end="")

torch.save(model.state_dict(), model_save_path)
print("\n\033[92mModel saved successfully\033[0m\n")
