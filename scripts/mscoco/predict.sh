python  train.py \
--config_name conica-clip \
--feature_path /dataset/caption/mscoco/features/ViT-L/14@336px \
--output_dir predict_clip \
--resume_from_checkpoint output/clip-vit_rl/checkpoints/checkpoint-8855  \
--do_predict \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4
