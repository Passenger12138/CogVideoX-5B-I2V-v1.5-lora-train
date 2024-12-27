# CogVideo & CogVideoX Finetune

## Project Updates
- 🔥🔥 **News**: ```2024/12/15```: 🚀 **We are excited to release the LoRA fine-tuning code for CogVideoX 1.5 by Diffusers**, designed specifically for image-to-video (image2video) tasks! This update brings significant improvements and new features to elevate your training experience. The full training startup code can be found in the `finetune` folder. Here's what's new:

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




