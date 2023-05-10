import pandas as pd
import numpy as np

from skimage.io import imread
from skimage.transform import resize

import torch
from torchvision.models import resnet50, ResNet50_Weights, resnet152
from tqdm import tqdm

import os

from ImageUtils import resize_img, filter_img
from Config import IMG_SIZE, N_OF_IMGS, N_EPOCHS

model = resnet50()
model.load_state_dict(torch.load(f"./models/model_{IMG_SIZE}x{IMG_SIZE}_{N_OF_IMGS}imgs_{N_EPOCHS}r.pt"))

test_csv = pd.read_csv("./test.csv")
test_csv.head()
out_csv = pd.read_csv("./out.csv")

imgs = []

# จำนวนรูปที่ต้องการ test
num_of_img = len(test_csv["id"].values)

print(f"\nLoading {num_of_img} images...\n")

for img_name in tqdm(test_csv["id"][:num_of_img]):
    
    img_path = os.path.join("./test", img_name.split("_")[0], img_name)

    # นำเข้ารูป
    img = imread(img_path, as_gray=True)

    filtered_img = filter_img(img)

    resized_img = resize_img(filtered_img, IMG_SIZE)

    imgs.append(resized_img)

test_imgs = np.array(imgs)

test_imgs = test_imgs.reshape(num_of_img, 3, IMG_SIZE, IMG_SIZE)
test_imgs = torch.from_numpy(test_imgs).to(torch.float32)

with torch.no_grad():
    output = model(test_imgs)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# out_csv['label'] = predictions
print(f"Calculating results...\n")
results = []
wrong_rate = num_of_img
for index, label in enumerate(tqdm(test_csv["label"][:num_of_img])):
    result = "wrong"
    if label == predictions[index]:
        result = "correct"
        wrong_rate -= 1
    results.append(result)

out_csv.loc[:num_of_img - 1, 'label'] = predictions
out_csv.loc[:num_of_img - 1, 'result'] = results
out_csv.head()

print(f"\nCorrect images {num_of_img - wrong_rate} of {num_of_img}")

print(f"Accuracy: {100 - (wrong_rate / num_of_img * 100)}%\n")

out_csv.to_csv("./out.csv", index=False)
