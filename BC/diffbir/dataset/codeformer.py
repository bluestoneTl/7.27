from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
import torch
from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        # file_list: str,
        file_list_HQ: str,
        file_list_LQ: str,        
        # file_list_condition: str,               # clip预先提取的图像特征
        # file_list_RGB: str,                     # 【融合RGB图像方法二】      
        file_list_edge: str,                    # 【融合边缘图】
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        # 以下这些参数原本用于生成LQ图像，现暂时保留
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        # self.file_list = file_list
        # self.image_files = load_file_list(file_list)
        self.file_list_HQ = file_list_HQ
        self.file_list_LQ = file_list_LQ
        # self.file_list_condition = file_list_condition
        # self.file_list_RGB = file_list_RGB
        self.file_list_edge = file_list_edge
        self.image_files_HQ = load_file_list(file_list_HQ)
        self.image_files_LQ = load_file_list(file_list_LQ)
        # self.image_files_condition = load_file_list(file_list_condition)    # clip预先提取的图像特征
        # self.image_files_RGB = load_file_list(file_list_RGB)
        self.image_files_edge = load_file_list(file_list_edge)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # 保留原有的HQ图像退化生成LQ图像的配置参数
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5) 
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")   
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image
    
    def load_condition_features(self, index: int) -> Optional[np.ndarray]:
        """加载条件特征文件"""
        condition_file = self.image_files_condition[index]
        condition_path = condition_file["image_path"] 
        # Tensor shape: torch.Size([1, 768])
        try:
            condition_features = torch.load(condition_path)  # 用这个实验
            return condition_features
        except Exception as e:
            print(f"Failed to load condition features from {condition_path}: {e}")
            return None
        
    def load_condition_features2(self, index: int) -> Optional[np.ndarray]:
        """加载条件特征文件"""
        condition_file = self.image_files_edge[index]
        condition_path = condition_file["image_path"] 
        # Tensor shape: torch.Size([1, 768])
        try:
            condition_features = torch.load(condition_path)  # 用这个实验
            return condition_features
        except Exception as e:
            print(f"Failed to load condition features from {condition_path}: {e}")
            return None

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
            # 加载HQ图像
            img_gt = None
            while img_gt is None:
                # 从HQ文件列表中获取对应图像路径
                image_file_HQ = self.image_files_HQ[index]
                gt_path = image_file_HQ["image_path"]
                prompt = image_file_HQ["prompt"]
                img_gt = self.load_gt_image(gt_path)
                if img_gt is None:
                    print(f"filed to load {gt_path}, try another image")
                    index = random.randint(0, len(self) - 1)    #保留意见len(self.image_files_HQ)

            # 加载LQ图像
            img_lq = None
            while img_lq is None:
                image_file_LQ = self.image_files_LQ[index]
                lq_path = image_file_LQ["image_path"]
                img_lq = self.load_gt_image(lq_path)
                if img_lq is None:
                    print(f"filed to load {lq_path}, try another image")
                    index = random.randint(0, len(self) - 1)

            # 加载RGB图像特征          # 【融合RGB图像方法二】
            # rgb = self.load_condition_features(index)

            # 加载RGB图像，用作引导
            # rgb = None
            # while rgb is None:
            #     image_file_RGB = self.image_files_RGB[index]
            #     rgb_path = image_file_RGB["image_path"]
            #     rgb = self.load_gt_image(rgb_path)
            #     if rgb is None:
            #         print(f"filed to load {rgb_path}, try another image")
            #         index = random.randint(0, len(self) - 1)
            # rgb = (rgb[..., ::-1] / 255.0).astype(np.float32)

            # 加载边缘图，用作引导
            edge = None
            while edge is None:
                image_file_edge = self.image_files_edge[index]
                edge_path = image_file_edge["image_path"]
                edge = self.load_gt_image(edge_path)
                if edge is None:
                    print(f"filed to load {edge_path}, try another image")
                    index = random.randint(0, len(self) - 1)
            edge = (edge[..., ::-1] / 255.0).astype(np.float32)

            # # 加载边缘图特征
            # edge = self.load_condition_features2(index)

            # 加载rgb图像特征，clip预先提取的图像特征
            # condition = self.load_condition_features(index)

            img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
            gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)

            lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)

            return gt, lq, prompt, edge


    def __len__(self) -> int:
        # return len(self.image_files)
        return len(self.image_files_HQ)
