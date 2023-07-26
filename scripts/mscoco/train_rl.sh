python  train.py \
--config_name conica-clip_rn \
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
--resume_from_checkpoint clip-vit_xe_checkpoint \
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