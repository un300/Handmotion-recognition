import os
import math
import cv2
from PIL import Image as im
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


def resize_img(src, size=160):
    # 1. resize
    if isinstance(src, str):
        img = np.array(im.open(src).convert('RGB'))
    elif isinstance(src, np.ndarray):
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


class VidoeGenerator(Dataset):
    def __init__(
            self,
            video_folder_path,
            label_csv_path,
            fps
    ):
        self.video_folder_path = video_folder_path

        indexes = pd.read_csv(
            label_csv_path,
            encoding='utf-8-sig'
        )
        indexes['file_path'] = [f'{self.video_folder_path}/{path}.mp4' for path in indexes['id']]

        self.indexes = indexes[['file_path', 'label']]
        self.fps = fps

    def __len__(self):
        return self.indexes.shape[0]

    def get_video(self, file_path):
        captures = cv2.VideoCapture(file_path)
        frames = []
        for _ in range(self.fps):
            _, img = captures.read()
            img = resize_img(img, size=160) / 255
            frames.extend([img])
        return torch.FloatTensor(
            np.array(frames)
        ).permute(3, 0, 1, 2)

    def __getitem__(self, idx):
        path, label = self.indexes.iloc[idx, :]
        frames = self.get_video(path)
        return frames, label


