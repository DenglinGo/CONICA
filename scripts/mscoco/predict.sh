python  train.py \
--config_name conica-clip_rn \
--feature_path /dataset/caption/mscoco/features/RN50x4 \
--output_dir predict_clip \
--resume_from_checkpoint output/clip-RN50x4/rl/checkpoints/checkpoint-8855  \
--do_predict \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4
