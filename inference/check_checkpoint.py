"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import warnings
import argparse
from typing import Literal

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video
import json
import os

def get_samplesize(input_img,bucket_size):
    success = False
    try:
        width, height = input_img.size 
        aspect_ratio = width / height
        new_height = int((bucket_size *bucket_size / aspect_ratio) ** 0.5) 
        new_width = int(new_height * aspect_ratio)
        new_height = new_height //16 * 16
        new_width = new_width //16 * 16
        sample_size = [new_height,new_width]
        success =True
        return success, sample_size
    except Exception as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument('--model_name', type=str, default="/maindata/data/shared/public/haobang.geng/haobang-huggingface/CogVideoX1.5-5B-I2V/", help='Path to the model')
    parser.add_argument('--results_dir1', type=str, default='/path/to/default/results', help='Directory to save the results')
    parser.add_argument('--lora_dir', type=str, default='/path/to/default/lora', help='Directory containing the LoRA weights')
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument('--json_file_path', type=str, default="/maindata/data/shared/public/haobang.geng/dataset/character-img/imgs_joycaption.json", help='Path to the JSON file containing image paths and captions')
    parser.add_argument('--video_length', type=int, default=49, help='Length of the generated video')
    parser.add_argument('--bucket_size', type=int, default=768, help='Length of the generated video')
    parser.add_argument('--lora_weight', type=float, default=1, help='Weight of the LoRA')
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    # 获取第一级目录下所有后缀为.safetensors的文件
    args = parser.parse_args()
    dtype = torch.bfloat16
    guidance_scale = 6.0
    num_inference_steps = 50
    num_frames = 49
    fps = 8
    generate_type = "i2v"
    results_dir1 = args.results_dir1
    os.makedirs(results_dir1, exist_ok=True)
    lora_dir = args.lora_dir
    
    # 获取第一级目录下所有保存checkpoints的文件
    subfolder_pattern = "checkpoint-"
    subfolders = [f for f in os.listdir(lora_dir) if subfolder_pattern in f]

    # 读取 JSON 文件中的图像路径和提示
    json_file_path = args.json_file_path
    with open(json_file_path, 'r', encoding='utf-8') as f: 
        img_data_list = json.load(f)[:10]

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_name, torch_dtype=dtype)
    # pipe = CogVideoXImageToVideoPipeline()
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to("cuda")

    for subfolder in subfolders:
        subfolder_path = os.path.join(lora_dir,subfolder)
        result_base_name = subfolder
        results_dir = os.path.join(results_dir1, result_base_name)
        os.makedirs(results_dir, exist_ok=True)

        pipe.load_lora_weights(subfolder_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-i2v-lora")
        pipe.set_adapters(["cogvideox-i2v-lora"], [args.lora_weight])
        
        for img_data in img_data_list:
            eval_img_path = img_data["img"]
            prompt = "A person sks gentle motion. " + img_data["caption"] + " The video is of high quality, High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
            image = load_image(image=eval_img_path)
            _,sample_size = get_samplesize(image,args.bucket_size)
            print(sample_size)

            video_generate = pipe(
                height=sample_size[0],
                width=sample_size[1],
                prompt=prompt,
                image=image,
                # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
                num_videos_per_prompt=1,  # Number of videos to generate per prompt
                num_inference_steps=num_inference_steps,  # Number of inference steps
                num_frames=num_frames,  # Number of frames to generate
                use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(args.seed),  # Set the seed for reproducibility
            ).frames[0]

            avatar_dynamic_img_name = os.path.basename(eval_img_path)
            base_name, _ = os.path.splitext(avatar_dynamic_img_name)
            # 将扩展名替换为 .mp4
            result_video_name = f"{base_name}.mp4"
            result_video_path = os.path.join(results_dir, result_video_name)
            export_to_video(video_generate, result_video_path, fps=fps)

        pipe.unload_lora_weights()


    
    