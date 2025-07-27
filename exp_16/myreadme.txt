python  inference.py 
--task denoise 
--upscale 1 
--version v2.1 
--captioner llava 
--cfg_scale 8 
--noise_aug 0 
--input inputs/demo/bid 
--output results/v2.1_demo_bid

单卡训练：

数据集处理
# collect all iamge files in img_dir
find datasets/data_ZZCX_singleHQ -type f > datasets/files.list
# shuffle collected files
shuf datasets/files.list > datasets/files_shuf.list
# pick train_size files in the front as training set
head -n 1300 datasets/files_shuf.list > datasets/files_shuf_train.list
# pick remaining files as validation set
tail -n 1301 datasets/files_shuf.list > datasets/files_shuf_val.list

test:
    find datasets/ZZCX_01_14/test/HQ -type f > datasets/ZZCX_01_14/test/HQ.list
    shuf datasets/ZZCX_01_14/test/HQ.list > datasets/ZZCX_01_14/test/HQ_shuf.list
    find datasets/ZZCX_01_14/test/LQ -type f > datasets/ZZCX_01_14/test/LQ.list
    shuf datasets/ZZCX_01_14/test/LQ.list > datasets/ZZCX_01_14/test/LQ_shuf.list
train:
    find datasets/ZZCX_01_14/train/HQ -type f > datasets/ZZCX_01_14/train/HQ.list
    shuf datasets/ZZCX_01_14/train/HQ.list > datasets/ZZCX_01_14/train/HQ_shuf.list
    find datasets/ZZCX_01_14/train/LQ -type f > datasets/ZZCX_01_14/train/LQ.list
    shuf datasets/ZZCX_01_14/train/LQ.list > datasets/ZZCX_01_14/train/LQ_shuf.list
val:
    find datasets/ZZCX_01_14/val/HQ -type f > datasets/ZZCX_01_14/val/HQ.list
    shuf datasets/ZZCX_01_14/val/HQ.list > datasets/ZZCX_01_14/val/HQ_shuf.list
    find datasets/ZZCX_01_14/val/LQ -type f > datasets/ZZCX_01_14/val/LQ.list
    shuf datasets/ZZCX_01_14/val/LQ.list > datasets/ZZCX_01_14/val/LQ_shuf.list

Stage 1:
    python train_stage1.py --config configs/train/train_stage1.yaml

Stage 2:
    Download pretrained Stable Diffusion v2.1 
    wget https://hf-mirror.com/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
    python train_stage2.py --config configs/train/train_stage2.yaml

测试命令：
python -u inference.py --task denoise --upscale 1 --version v2 --sampler spaced --steps 50 --captioner none --pos_prompt '' --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' --cfg_scale 4.0 --input datasets/ZZCX_01_14/test/LQ --output results/1_7_1 --device cuda --precision fp32

自定义模型的测试命令模板：
python -u inference.py \
--upscale 1 \
--version custom \
--train_cfg configs/train/train_stage2.yaml \
--ckpt experiment/experiment_1/stage2/checkpoints/0030000.pt \
--captioner none \
--cfg_scale 4.0 \
--noise_aug 0 \
--input datasets/ZZCX_2_1/test/LQ \
--condition_path datasets/ZZCX_2_1/test/condition_swinir_LQ \
--output results/2.26/custom_5 \
--precision fp32 \
--sampler spaced \
--steps 50 \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' 

推理实验  
1_7     都是用自定义模型测试,  --captioner none
1_7_2   都是用自定义模型测试   --captioner llava
1_7_3   v1的denoise测试  
1_7_4   v2的denoise测试    更改了bid_loop.py的v2加载模型为 swinir

【注意，去噪命令需要在pretrained_models.py中更改模型的加载路径，在common.py中更改加载方式】
去噪案例命令模板：
python -u inference.py \
--task denoise \
--upscale 1 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner none \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4.0 \
--input datasets/ZZCX_01_14/test/LQ \
--output results/1.13/denoise_0 \
--device cuda \
--precision fp32

【1.12推理实验】    在results/1.12 下       
custom_0           自定义模型测试 --precision fp16   全黑
custom_1           自定义模型测试 --precision fp32   出现图片，但奇怪质量
custom_2           测试训练集     --precision fp32   出现图片，但奇怪质量，同上
denoise_0          去噪案例命令   --precision fp16   复原效果差
denoise_1          去噪案例命令   --precision fp32   复原效果差，同上，【为什么这个fp32和fp16相同呢，而且fp16不是全黑】
denoise_2          测试训练集     --precision fp32   复原效果差，同上
denoise_3          直接使用他们给出的去噪模型测试     效果好，背景稍微不对，应该是不经过训练的问题

【1.13推理实验】    在results/1.13 下       
custom_0           自定义模型测试  --precision fp32   奇怪的表现，但看零件形状是完整的，要去看一下两个推理方式哪里不同
custom_1           修复推理bug     --precision fp32   成功！，问题是脚本加载的时候，权重处理问题
custom_2           修复后，测试    --precision fp16   成功！全黑的问题是bitsandbytes版本冲突，但是不报错正常运行了
denoise_0          去噪案例命令    --precision fp32   哈哈，效果很好！稍微的缺点，1k的背景暗一些

【1.14推理实验】    在results/1.14 下       
custom_0           3w次训练测试    --precision fp32    
custom_1           1w次训练测试    --precision fp32   
custom_2           2w次训练测试    --precision fp32  
custom_3           mini_test      --cfg_scale 1.0     --strength 1.0
custom_4           mini_test      --cfg_scale 2.0     --strength 1.0
custom_5           mini_test      --cfg_scale 4.0     --strength 1.0
custom_6           mini_test      --cfg_scale 6.0     --strength 1.0
custom_7           mini_test      --cfg_scale 8.0     --strength 1.0
custom_8           mini_test      --cfg_scale 4.0     --strength 0.25
custom_9           mini_test      --cfg_scale 4.0     --strength 0.5
custom_10          mini_test      --cfg_scale 4.0     --strength 1.0
custom_11          mini_test      --cfg_scale 4.0     --strength 2.0
custom_12          mini_test      --cfg_scale 4.0     --strength 4.0


find datasets/ZZCX_01_20/train/HQ -type f > datasets/ZZCX_01_20/train/HQ.list
find datasets/ZZCX_01_20/train/LQ -type f > datasets/ZZCX_01_20/train/LQ.list
find datasets/ZZCX_01_20/train/condition_RGB -type f > datasets/ZZCX_01_20/train/condition_RGB.list
find datasets/ZZCX_01_20/train/condition_edge -type f > datasets/ZZCX_01_20/train/condition_edge.list

find datasets/ZZCX_2_1/train/HQ -type f > datasets/ZZCX_2_1/train/HQ.list
find datasets/ZZCX_2_1/train/LQ -type f > datasets/ZZCX_2_1/train/LQ.list
find datasets/ZZCX_2_1/train_RGB/HQ -type f > datasets/ZZCX_2_1/train_RGB/RGB.list
find datasets/ZZCX_2_1/train_RGB/condition_HQ -type f > datasets/ZZCX_2_1/train_RGB/condition_RGB.list
find datasets/ZZCX_2_1/train_RGB/condition_edge_HQ -type f > datasets/ZZCX_2_1/train_RGB/condition_edge.list


find datasets/ZZCX_2_1/train/condition_swinir_LQ -type f > datasets/ZZCX_2_1/train/condition_swinir_LQ.list

find datasets/ZZCX_2_1/test/LQ -type f > datasets/ZZCX_2_1/test/LQ.list
find datasets/ZZCX_2_1/test/HQ -type f > datasets/ZZCX_2_1/test/HQ.list
