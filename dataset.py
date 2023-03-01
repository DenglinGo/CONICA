# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:46
# @Author : Lingo
# @File : dataset.py
from torch.utils.data import Dataset
import numpy as np
import random
import os.path as op
import pandas as pd
from torchvision.transforms import Compose, Resize, RandomResizedCrop, CenterCrop, ToTensor, Normalize, \
    InterpolationMode, RandomHorizontalFlip
from PIL import Image
import string


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def valid_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ImageCaptionDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 df: pd.DataFrame,
                 feature_type: str = "clip",
                 seq_per_img: int = 5,
                 training=True,
                 max_feats_seq_len=512
                 ):
        self.file_path = file_path
        self.feature_type = feature_type
        self.df = df
        self.training = training
        self.seq_per_img = seq_per_img
        self.max_feats_seq_len = max_feats_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        gts_caps = eval(item['sentences'])
        img_feats = np.load(op.join(self.file_path, str(item["id"]) + ".npy"))
        num_caps = len(gts_caps)
        feats_len = len(img_feats)
        cls_feats = None
        if self.feature_type in ["clip", "vit"]:
            cls_feats = img_feats[:1]
            img_feats = img_feats[1:]
        elif self.feature_type in ["vinvl", "butd", "cnn"]:
            cls_feats = img_feats.mean(0)[None, :]
            if feats_len > self.max_feats_seq_len:
                img_feats = img_feats[:self.max_feats_seq_len]
            else:
                pad_feats = np.zeros((self.max_feats_seq_len - feats_len, img_feats.size(1)),dtype=np.float32)
                img_feats = np.concatenate((img_feats, pad_feats), axis=0)
        sample_caps = ["" for _ in range(self.seq_per_img)]
        if self.training:
            np.random.shuffle(img_feats)
            img_feats = img_feats[:self.max_feats_seq_len]
            if num_caps < self.seq_per_img:
                sample_caps[:num_caps] = gts_caps
                for _ in range(self.seq_per_img - num_caps):
                    sample_caps[num_caps + _] = gts_caps[random.randint(0, num_caps - 1)]
            else:
                _ = random.randint(0, num_caps - self.seq_per_img)
                sample_caps = gts_caps[_:_ + self.seq_per_img]

        img_feats = np.concatenate((cls_feats, img_feats), axis=0)
        return img_feats, sample_caps, gts_caps


class ImageDataset(Dataset):
    def __init__(self,
                 root: str,
                 df: pd.DataFrame,
                 seq_per_img: int = 1,
                 transform=None,
                 training=False
                 ):
        self.root = root
        self.df = df
        self.seq_per_img = seq_per_img
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        gts_caps = eval(item['sentences'])
        img = Image.open(op.join(self.root, item['filepath'], item['filename']))
        if self.transform:
            img = self.transform(img)
        num_caps = len(gts_caps)
        sample_caps = ["" for _ in range(self.seq_per_img)]
        if self.training:
            if num_caps < self.seq_per_img:
                sample_caps[:num_caps] = gts_caps
                for _ in range(self.seq_per_img - num_caps):
                    sample_caps[num_caps + _] = gts_caps[random.randint(0, num_caps - 1)]
            else:
                _ = random.randint(0, num_caps - self.seq_per_img)
                sample_caps = gts_caps[_:_ + self.seq_per_img]
        return img.numpy(), sample_caps, gts_caps
