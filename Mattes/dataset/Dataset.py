# Adjusted from Source: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/dataset.py
#

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from medmnist import INFO
import torchvision.transforms as transforms


class MedMNIST(Dataset):

    def __init__(
        self,
        data_path,
        split,
        transform=None,
        target_transform=None,
        as_rgb=False,
    ):
        """
        Args:
            data_path (str): Path to the .npz file containing data, required
            split (string): 'train', 'val' or 'test', required
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
            as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
        """

        self.size = 28
        self.size_flag = ""

        self.info = INFO["dermamnist"]

        npz_file = np.load(data_path)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split in ["train", "val", "test"]:
            self.imgs = npz_file[f"{self.split}_images"]
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def __len__(self):
        assert self.info["n_samples"][self.split] == self.imgs.shape[0]
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.size}"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class MedMNIST2D(MedMNIST):

    def __getitem__(self, index):
        """
        return: (without transform/target_transform)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        canny_img = torch.from_numpy(cv2.Canny(img, 50, 150)).to(torch.float)
        canny_img = canny_img.repeat(3,1,1)
        
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        target = torch.from_numpy(target).to(torch.long)
        target = torch.nn.functional.one_hot(target.flatten(), 7).squeeze().to(torch.float32)

        return img, canny_img, target