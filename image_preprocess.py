import math
import cv2
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def resize_img(src, size=160):
    # 1. resize
    if isinstance(src, str):
        img = np.array(im.open(src).convert('RGB'))
    elif isinstance(src, np.array):
        img = src

    h, w = img.shape[:2]
    ratio = min(size / w, size / h)

    tw, th = (int(w * ratio), int(h * ratio))

    img = cv2.resize(
        img, dsize=(tw, th),
        interpolation=cv2.INTER_AREA
    )

    dx1 = dx2 = dy1 = dy2 = 0
    # 2. padding
    if tw < th:
        dx1 = (size - tw) // 2
        dx2 = (size - tw) // 2 + math.ceil((size - tw) % 2)
    else:
        dy1 = (size - th) // 2
        dy2 = (size - th) //2 + math.ceil((size - th) % 2)

    img = cv2.copyMakeBorder(
        img, dy1, dy2, dx1, dx2,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return img

# %%