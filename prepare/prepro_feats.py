# -*- coding: utf-8 -*-
# @Time : 2022/6/1 8:21
# @Author : Lingo
# @File : prepro_feats.py
# @Descripition : runs only on single GPU

from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import clip
from clip.clip import _transform
from clip.model import VisionTransformer, ModifiedResNet
import os.path as op
from PIL import Image
import timm
import warnings
import argparse
import os
import numpy as np
from tqdm import tqdm
from timm.models.vision_transformer import resize_pos_embed


class ImageDataset(Dataset):
    def __init__(self, root_path, pre_process):
        self.root_path = root_path
        self.imgs = os.listdir(root_path)
        self.preprocess = pre_process

    def __len__(self):
        self.imgs.sort()
        return len(self.imgs)

    def __getitem__(self, idx):
        file_name = self.imgs[idx]
        if self.preprocess:
            return file_name, self.preprocess(Image.open(op.join(self.root_path, self.imgs[idx])))
        return file_name, Image.open(op.join(self.root_path, self.imgs[idx]))


class ImageFeatureProjection(nn.Module):
    def __init__(self, model_name, input_r, output_r, patch_l):
        super().__init__()
        num_patches = (input_r // patch_l) ** 2 if patch_l is not None else (input_r // 32) ** 2
        if model_name in clip.available_models():
            clip_model, _ = clip.load(model_name)
            clip_model.to(device='cpu', dtype=torch.float)
            self.visual = clip_model.visual
        elif model_name in timm.list_models():
            self.visual = timm.create_model(model_name, pretrained=True)
        else:
            raise NotImplementedError
        self.global_pool = None
        if isinstance(self.visual, ModifiedResNet):
            positional_embedding = nn.Parameter(
                torch.zeros(1, 1 + num_patches, self.visual.attnpool.positional_embedding.size(-1)))
            resized_positional_embedding = resize_pos_embed(self.visual.attnpool.positional_embedding.unsqueeze(0),
                                                            positional_embedding)
            self.global_pool = self.visual.attnpool
            self.global_pool.positional_embedding = nn.Parameter(resized_positional_embedding.squeeze(0), )
            self.visual.attnpool = nn.Identity()
        elif isinstance(self.visual, VisionTransformer):
            positional_embedding = nn.Parameter(
                torch.zeros(1, 1 + num_patches, self.visual.positional_embedding.size(-1)))
            resized_positional_embedding = resize_pos_embed(self.visual.positional_embedding.unsqueeze(0),
                                                            positional_embedding)
            self.visual.positional_embedding = nn.Parameter(resized_positional_embedding.squeeze(0), )
        elif isinstance(self.visual, timm.models.ResNet):
            self.global_pool = self.visual.global_pool
            self.visual.global_pool = nn.Identity()
            self.visual.fc = nn.Identity()
        elif isinstance(self.visual, timm.models.efficientnet.EfficientNet):
            self.global_pool = nn.Sequential(
                self.visual.conv_head,
                self.visual.global_pool
            )
            self.visual.conv_head = nn.Identity()
            self.visual.global_pool, self.visual.classifier = nn.Identity(), nn.Identity()
        elif isinstance(self.visual, timm.models.SwinTransformer):
            self.global_pool = self.visual.avgpool
            self.visual.avgpool = nn.Identity()
            self.visual.head = nn.Identity()
        else:
            raise NotImplementedError

        self.pooling = nn.AdaptiveAvgPool2d((output_r, output_r)) if output_r!=input_r//patch_l else nn.Identity()

    @torch.no_grad()
    def forward(self, x):
        if isinstance(self.visual, ModifiedResNet):
            x = self.visual(x)
            g = self.global_pool(
                x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1))  # NCHW -> (HW)NC

        elif isinstance(self.visual, VisionTransformer):
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            grid = x.shape[-1]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                       device=x.device), x],
                dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            g = self.visual.ln_post(x[:, 0, :])  # ND
            if self.visual.proj is not None:
                g = g @ self.visual.proj
            x = x[:, 1:, :].permute(0, 2, 1)
            x = x.reshape(x.shape[0], -1, grid, grid)  # [N,width,grid,grid]


        elif isinstance(self.visual, (timm.models.ResNet, timm.models.efficientnet.EfficientNet)):
            x = self.visual(x)
            g = self.global_pool(x)  # ND1

        elif isinstance(self.visual, timm.models.SwinTransformer):
            x = self.visual.patch_embed(x)
            if self.visual.absolute_pos_embed is not None:
                x = x + self.visual.absolute_pos_embed
            x = self.visual.pos_drop(x)
            x = self.visual.layers(x)  # NLD
            g = self.global_pool(x.transpose(1, 2))
            g = torch.flatten(g, 1)


        else:
            raise NotImplementedError
        x = self.pooling(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)
        return g.unsqueeze(1), x


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-input_resolution', type=int, default=336, help='')
    parser.add_argument('-output_resolution', type=int, default=24, help='')
    parser.add_argument('-patch_length', type=int, default=14, help='')
    parser.add_argument('-model_name', type=str, default='ViT-L/14@336px', help='')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-dataset', type=str, default='/dataset/caption/mscoco/images/val2014',
                        help='path of dataset')
    parser.add_argument('-save_path', type=str, default='/dataset/caption/mscoco/features/', help='path of dataset')
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.gpu else "cpu")
    prepocess = _transform(args.input_resolution)
    dataset_path = args.dataset
    save_path = args.save_path
    model_name = args.model_name

    proj = ImageFeatureProjection(model_name, input_r=args.input_resolution, output_r=args.output_resolution,
                                  patch_l=args.patch_length).to(device)
    proj.eval()
    dataset = ImageDataset(dataset_path, prepocess)
    if not op.exists(op.join(save_path, model_name)):
        os.makedirs(op.join(save_path, model_name))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=False)
    with torch.no_grad():
        for filenames, imgs in tqdm(dataloader):
            features = proj(imgs.to(device))
            for i, filename in enumerate(filenames):
                id = int(filename.split('.')[0].split('_')[-1])

                g_feature, l_features = features[0][i], features[1][i]
                np.savez_compressed(op.join(args.save_path, model_name, str(id)),
                                    l_features=l_features.cpu().float().numpy(),
                                    g_feature=g_feature.cpu().float().numpy())
