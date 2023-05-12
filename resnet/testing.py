import torch
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

from helpers.transform import transform

num_classes = 10

model_path = "../models/AlexNet_v2_augmented.pt"

test_dataset = ImageFolder(root="../datasets/test", transform=transform)

print(f"\nLoaded \033[94m{len(test_dataset)}\033[0m dataset to model\n")

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = resnet50()

model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(model_path))
model.eval()

class_labels = [
    "apple",
    "avocado",
    "banana",
    "cherry",
    "kiwi",
    "mango",
    "orange",
    "pineapple",
    "strawberries",
    "watermelon"
]

images_length = len(test_dataset)
correct_images_length = 0

true_classes = {
    "apple": 0,
    "avocado": 0,
    "banana": 0,
    "cherry": 0,
    "kiwi": 0,
    "mango": 0,
    "orange": 0,
    "pineapple": 0,
    "strawberries": 0,
    "watermelon": 0
}

predict_classes = {
    "apple": 0,
    "avocado": 0,
    "banana": 0,
    "cherry": 0,
    "kiwi": 0,
    "mango": 0,
    "orange": 0,
    "pineapple": 0,
    "strawberries": 0,
    "watermelon": 0
}

model_file_name = os.path.basename(model_path)

print(f"Start testing \033[94m{model_file_name}\033[0m model...\n")

print("\033[93m", end="")
for images, labels in tqdm(test_loader):
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    predicted_label = class_labels[predicted.item()]

    true_label = class_labels[labels.item()]

    result = int(predicted_label == true_label)

    correct_images_length += result

    true_classes[true_label] += 1
    predict_classes[true_label] += result
print("\033[0m", end="")

print()
for key in true_classes:
    print(f"{key.capitalize().ljust(12)} \t{(predict_classes[key] / true_classes[key] * 100):.2f}%")
print()

print(f"Overall accuracy is \033[92m{(correct_images_length / images_length * 100):.2f}%\033[0m\n")
