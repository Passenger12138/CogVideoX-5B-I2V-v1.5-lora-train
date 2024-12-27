# CogVideoX diffusers Fine-tuning Guide

## Project Updates

🚀 **We are excited to release the LoRA fine-tuning code for CogVideoX 1.5 by Diffusers**, designed specifically for image-to-video (image2video) tasks! This update brings significant improvements and new features to elevate your training experience. The full training startup code can be found in the `finetune` folder. Here's what's new:

### 🔥 Key Features & Improvements:
1. **💥 Bucket-based Multi-Resolution Training**: Unlock unparalleled model adaptability and performance across videos of all resolutions. This groundbreaking feature boosts the model’s ability to handle diverse video qualities with ease!
   
2. **⚡ Fixed RoPE (Relative Position Encoding) Configuration Error**: We’ve optimized the position encoding mechanism, solving the error in the original CogVideo code, resulting in smoother training and higher-quality outputs. No more misconfigurations—just pure efficiency!

3. **🔧 Corrected OFS Embedding Issue**: Previously, OFS embedding was incorrectly set to None in the original code. Now, it’s properly configured for stability and precision, improving the overall reliability and robustness of the model.

### ✅ Summary of Fixes:
- **Multi-resolution Support** 🖼️
- **Accurate Positional Encoding** 📍
- **Correct OFS Embedding Setup** 🔑
- **Optimized Multi-GPU Fine-Tuning** 💻

With these changes, you can now start training with **just one command**—simple, fast, and effective! 




🚀 **我们激动地宣布发布了基于 Diffusers 的 LoRA 微调代码，用于 CogVideoX 1.5，并支持图像到视频（image2video）任务！** 本次更新带来了重大的改进和新特性，训练启动代码可以在 `finetune` 文件夹中找到。以下是主要的新功能：

### 🔥 **关键功能与改进：**
1. **💥 基于桶的多分辨率训练**：释放前所未有的模型适应性和性能，适用于各种分辨率的视频。此项突破性功能增强了模型对不同视频质量的处理能力！
   
2. **⚡ 修复了 RoPE（相对位置编码）配置错误**：我们优化了位置编码机制，解决了原始 CogVideo 代码中的错误，从而提高了训练效率和输出质量。不再有配置错误——只剩下高效训练！

3. **🔧 解决了 OFS 嵌入设置为 None 的问题**：原始代码中，OFS 嵌入错误地设置为 None。现在它已经正确配置，提升了模型的稳定性和可靠性，确保了模型的精准性。

### ✅ **修复汇总：**
- **多分辨率支持** 🖼️
- **准确的位置编码** 📍
- **正确的 OFS 嵌入设置** 🔑
- **优化的多卡微调** 💻

通过这些改进，您现在只需**一条命令**就能开始训练——简单、快速、有效！



## Hardware Requirements

+ CogVideoX-5B-I2V 1 * A100

## Install Dependencies

Since the related code has not been merged into the diffusers release, you need to base your fine-tuning on the
diffusers branch. Please follow the steps below to install dependencies:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers # Now in Main branch
pip install -e .
```

## data preprocessing


We fellow [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun) data preprocessing method, They have provided a simple demo of training the Lora model through image data, which can be found in the [wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora) for details.

A complete data preprocessing link for long video segmentation, cleaning, and description can refer to [README](cogvideox/video_caption/README.md) in the video captions section. 

If you want to train a text to image and video generation model. You need to arrange the dataset in this format.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

The json_of_internal_datasets.json is a standard JSON file. The file_path in the json can to be set as relative path, as shown in below:
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

You can also set the path as absolute path as follow:
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

## Video DiT training
 
If the data format is relative path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

If the data format is absolute path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

Then, we run scripts/train.sh.
```sh
sh scripts/train_cogvideox_i2v_lora_single_rank.sh
```

We can choose whether to use deep speed in CogVideoX-Fun, which can save a lot of video memory. 

Some parameters in the sh file can be confusing, and they are explained in this document:

- `enable_bucket` is used to enable bucket training. When enabled, the model does not crop the images and videos at the center, but instead, it trains the entire images and videos after grouping them into buckets based on resolution.
- `random_frame_crop` is used for random cropping on video frames to simulate videos with different frame counts.
- `random_hw_adapt` is used to enable automatic height and width scaling for images and videos. When random_hw_adapt is enabled, the training images will have their height and width set to image_sample_size as the maximum and video_sample_size as the minimum. For training videos, the height and width will be set to video_sample_size as the maximum and min(video_sample_size, 512) as the minimum.
- `training_with_video_token_length` specifies training the model according to token length. The token length for a video with dimensions 512x512 and 49 frames is 13,312.
  - At 512x512 resolution, the number of video frames is 49;
  - At 768x768 resolution, the number of video frames is 21;
  - At 1024x1024 resolution, the number of video frames is 9;
  - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.

CogVideoX-5B-I2V-v1.5 without deepspeed:

```sh
export MODEL_PATH="./models/CogVideoX1.5-5B-I2V/"
export CACHE_PATH="./cache/"
export OUTPUT_PATH="output_dir/lora-v1"
export DATASET_META_NAME="./dataset/motion-dataset/train.json"
export DATASET_NAME=""

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT_MS=1000000  # 设置为 10min，或根据需要增加


accelerate launch train_cogvideox_image_to_video_lora.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1280 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --random_hw_adapt \
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

```


CogVideoX-5B-I2V-v1.5 with deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-2b-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --use_deepspeed --deepspeed_config_file scripts/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train_lora.py \
   --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1280 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --random_hw_adapt \
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
```
