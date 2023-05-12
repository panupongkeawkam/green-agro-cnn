import torch
from torchvision.models import alexnet
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

from tqdm import tqdm
import os

from helpers.transform import transform

num_classes = 10

model_path = "../models/AlexNet_v3_25epc.pt"

test_dataset = ImageFolder(root="../datasets/test", transform=transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"\nLoaded \033[94m{len(test_dataset)}\033[0m dataset to model\n")

predicted_folder = "./predicted"

if os.path.exists(predicted_folder):
    # ล้างข้อมูล predict เก่าทั้งหมด
    for root, dirs, files in os.walk(predicted_folder, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

    os.rmdir(predicted_folder)

model = alexnet()

num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, num_classes)

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

image_count = 0

for images, labels in tqdm(test_loader):
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    predicted_label = class_labels[predicted.item()]

    true_label = class_labels[labels.item()]

    result = int(predicted_label == true_label)

    correct_images_length += result

    true_classes[true_label] += 1
    predict_classes[true_label] += result

    dir_path = os.path.join("./predicted", true_label, "correct" if result == 1 else "wrong")

    src_image_file_path = test_dataset.imgs[image_count][0]

    # จำแนกและบันทึกภาพที่ถูกหรือผิด
    for image, label in zip(images, labels):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        src_image = Image.open(src_image_file_path).convert('RGB')

        file_name = os.path.basename(src_image_file_path)
        file_path = os.path.join(dir_path, f"{predicted_label}_{image_count}.jpeg")

        src_image.save(file_path)
        src_image.close()

    image_count += 1
print("\033[0m", end="")

print()
for key in true_classes:
    print(f"{key.capitalize().ljust(12)} \t{(predict_classes[key] / true_classes[key] * 100):.2f}%")
print()

print(f"Overall accuracy is \033[92m{(correct_images_length / images_length * 100):.2f}%\033[0m\n")
