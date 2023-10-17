from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import csv
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import train_test_split

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, df, mode="train"):
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomVerticalFlip(p=0.5),
                                                 tv.transforms.RandomHorizontalFlip(p=0.5), tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(train_mean, train_std)])
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        output_list = self.df.values

        label = np.zeros((2), dtype=int)
        label[0] = int(output_list[idx][1]) #'crack'
        label[1] = int(output_list[idx][2]) #'inactive'

        image = imread(output_list[idx][0]) #filename
        image = gray2rgb(image)
        image = self._transform(image)
        label = torch.from_numpy(label)
        return image, label