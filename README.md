# CONICA: A Contrastive Image Captioning Framework with Robust Similarity Learning
This is the code implementation for the paper titled: "CONICA: A Contrastive Image Captioning Framework with Robust Similarity Learning" (Accepted to ACM MM 2023)

# Requirements
+ python>=3.8
+ pytorch=1.11.0 & torchvision=0.12.0
+ transformers=4.29.1
+ tokenizers=0.13.3
+ clip=1.0
+ other packages: pycocotools, pycocoevalcap,tqdm,pandas,tensorboard and timm

# Useage
## 1.Preparation
### Features
``` 
python prepare/prepro_feats.py -model_name ViT-L/14@336px -input_resolution 336 -dataset “your path to dataset(mscoco or others)” 
``` 

### Dataset
``` 
python prepare/prepro_datasets.py -karpathy_split_json “your path to karpathy split” -output_file /dataset/mscoco.csv
``` 
You can donwload karpathy split json from: [this link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

### HFConfiguration & Tokenizers 
``` 
 python prepare/prepro_conf_tokenizer.py -dataset_path /dataset/mscoco.csv -output_file conica-clip
``` 
### Words Frequency
```
python prepare/prepro_ngrams.py -input_csv /dataset/caption/mscoco.csv -output_pkl /dataset/caption/cache_document_frequency/coco-train-words.p
```

## 2. Training 
### 2.1. XE stage
```
python train.py \
--config_name conica-clip \
--max_features_len 256 \
--feature_path /dataset/caption/mscoco/features/ViT-L/14@336px \
--output_dir output/clip-vit_xe/checkpoints \
--do_train \
--evaluation_strategy epoch \
--logging_strategy steps \
--logging_steps 100 \
--logging_dir output/clip-vit_xe/logs \
--save_strategy epoch \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--weight_decay 1e-2 \
--num_train_epochs 30 \
--lr_scheduler_type linear \
--warmup_ratio 0.1 \
--fp16 \
--gradient_checkpointing \
--dataloader_pin_memory true \
--dataloader_num_workers 8
```
Then choosing the checkpoint with the highest cider for RL stage training. 

Or you can download the XE checkpoint from [this Google Drive Link](https://drive.google.com/file/d/1OCuThoS_3AOM6j1iDpFk0QX9ZmG2eYhh/view?usp=drive_link)

### 2.2. RL stage
```
python  train.py \
--config_name conica-clip \
--max_features_len 256 \
--feature_path /dataset/caption/mscoco/features/ViT-L/14@336px \
--output_dir output/clip-vit_rl/checkpoints \
--scst \
--init_tau \
--scst_num_sample_sequences 5 \
--do_train \
--evaluation_strategy epoch \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--resume_from_checkpoint "xe-checkpoint" \
--gradient_accumulation_steps 2 \
--learning_rate 5e-6 \
--weight_decay 1e-2 \
--num_train_epochs 20 \
--lr_scheduler_type constant \
--warmup_steps 0 \
--logging_strategy steps \
--logging_steps 100 \
--logging_dir output/clip-vit_rl/logs \
--save_strategy epoch \
--fp16 \
--dataloader_num_workers 12 \
--dataloader_pin_memory
```
You can download the RL checkpoint from [this Google Drive Link](https://drive.google.com/file/d/1QduIIlhG77v1uAnXPjjHJ9i8jhojgb2I/view?usp=drive_link)


## 3. Test
```
python  train.py \
--config_name conica-clip \
--feature_path /dataset/caption/mscoco/features/ViT-L/14@336px \
--output_dir predict_clip \
--resume_from_checkpoint "rl-checkpoint"  \
--do_predict \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4
```

# Citations
Please consider citing this paper if you use this code
```
@inproceedings{
author = {Deng, Lin and Zhong, Yuzhong and Wang, Maoning  and Zhang,Jianwei},
title = {CONICA: A Contrastive Image Captioning Framework with Robust Similarity Learning},
year = {2023},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {5109-5119}
}
```