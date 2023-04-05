# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:46
# @Author : Lingo
# @File : dataset.py
from torch.utils.data import Dataset
import numpy as np
import random
import os.path as op
import pandas as pd




class ImageCaptionDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 df: pd.DataFrame,
                 seq_per_img: int = 5,
                 training=True,
                 ):
        self.file_path = file_path
        self.df = df
        self.training = training
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        gts_caps = eval(item['sentences'])
        img_feats = np.load(op.join(self.file_path, str(item["id"]) + ".npz"))
        num_caps = len(gts_caps)
        cls_feat = img_feats["g_feature"]
        local_feat = img_feats["l_features"]
        sample_caps = ["" for _ in range(self.seq_per_img)]
        if self.training:
            if num_caps < self.seq_per_img:
                sample_caps[:num_caps] = gts_caps
                for _ in range(self.seq_per_img - num_caps):
                    sample_caps[num_caps + _] = gts_caps[random.randint(0, num_caps - 1)]
            else:
                _ = random.randint(0, num_caps - self.seq_per_img)
                sample_caps = gts_caps[_:_ + self.seq_per_img]

        return cls_feat, local_feat, sample_caps, gts_caps
