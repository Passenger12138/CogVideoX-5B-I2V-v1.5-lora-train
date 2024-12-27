#!/bin/bash

export MODEL_PATH="./models/CogVideoX1.5-5B-I2V/"
export CACHE_PATH="./cache/"
export OUTPUT_PATH="output_dir/lora-v1"
export DATASET_META_NAME="./dataset/motion-dataset/train.json"
export DATASET_NAME="./dataset/motion-dataset/"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT_MS=1000000  # 设置为 10min，或根据需要增加

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch train_cogvideox_image_to_video_lora.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --video_repeat 1 \
  --mixed_precision bf16 \
  --validation_epochs 100 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpoint_epochs 1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --training_with_video_token_length \
  --enable_bucket \
  --lora_dropout 0.0 \
  --nccl_timeout $NCCL_TIMEOUT_MS
