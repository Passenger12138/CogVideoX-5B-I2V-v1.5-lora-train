python cli_demo.py \
--prompt "A person smile" \
--image_or_video_path /maindata/data/shared/public/haobang.geng/dataset/1224-imgs/10523.png \
--model_path "/maindata/data/shared/public/haobang.geng/haobang-huggingface/CogVideoX1.5-5B-I2V/" \
--output_path ./output/debug2.mp4 \
--num_frames 17 \
--width 512 \
--height 768 \
--generate_type i2v \
--lora_path "/maindata/data/shared/public/haobang.geng/code/video-generate/CogVideo/finetune/output_dir/debug/checkpoint-2" \
--lora_rank 128

