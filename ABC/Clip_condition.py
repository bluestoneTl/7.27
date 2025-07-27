import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# python Clip_condition.py

# 加载CLIP模型，更换为ViT-L/14模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 定义图片目录
image_dir = "datasets/ZZCX_2_1/test/swinir_LQ"  

# 定义保存特征的目录
feature_dir = "datasets/ZZCX_2_1/test/condition_swinir_LQ"
os.makedirs(feature_dir, exist_ok=True)

# 遍历图片目录
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        # 读取图片
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        # 使用CLIP模型提取特征
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # 保存特征
        feature_filename = os.path.splitext(filename)[0] + ".pt"
        feature_path = os.path.join(feature_dir, feature_filename)
        torch.save(image_features, feature_path)
        print(f"{filename} 特征提取完成。")

print("图片特征提取并保存完成。")