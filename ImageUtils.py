from skimage.transform import rescale, resize
from skimage.filters import threshold_otsu
from skimage import exposure
import matplotlib.pyplot as plt

import numpy as np
import cv2

def resize_img(img: list, dim: int):
    # ความกว้างและสูงของรูปต้นฉบับ
    img_width = img.shape[0]
    img_height = img.shape[1]

    # หา shape ที่กว้างที่สุด
    max_dim = max(img_width, img_height)

    # scale factor ที่ต้องย่อหรือขยาย
    scale_factor = abs(dim / max_dim)

    # ทำการ scale รูปภาพ
    scaled_img = rescale(img, scale_factor, channel_axis=2)

    # ความกว้างและสูงของรูปที่ scale แล้ว
    scaled_img_width = scaled_img.shape[0]
    scaled_img_height = scaled_img.shape[1]

    # คำนวนหาขนาด padding
    pad_horizontal = (dim - scaled_img_width)
    pad_vertical = (dim - scaled_img_height)

    # กำหนด padding ให้ถูกด้าน
    padding = ((pad_horizontal // 2, pad_horizontal // 2), (pad_vertical // 2, pad_vertical // 2), (0, 0))

    # เติม padding ให้รูปภาพ
    padded_img = np.pad(scaled_img, padding, mode="constant")
    
    # ปรับขนาดเล็กน้อยกรณีเกิดกรณีพิเศษ เช่น รูปเป็นขนาด 255x256
    resized_img = resize(padded_img, (dim, dim))

    # norm pixel และแปลงเป็น float
    resized_img /= 255.0
    resized_img = resized_img.astype("float32")
    
    return resized_img

def filter_img(img: list):
    thresh = threshold_otsu(img)

    bin_img = img > thresh
    bin_img = bin_img.astype(np.float32)

    line_filt_diagonal = np.array([
        [0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0]
    ])

    edged_img = cv2.filter2D(img, -1, line_filt_diagonal, borderType=0)

    gamma_img = exposure.adjust_gamma(img, 5)

    merged_img = np.stack((bin_img, edged_img, gamma_img), axis=2)

    return merged_img
