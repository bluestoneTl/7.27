# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import os
import math
from inspect import isfunction
import torch
import torch.nn as nn
import numpy as np
from einops import repeat


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# class CheckpointFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
#                                    "dtype": torch.get_autocast_gpu_dtype(),
#                                    "cache_enabled": torch.is_autocast_cache_enabled()}
#         with torch.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with torch.enable_grad(), \
#                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             output_tensors = ctx.run_function(*shallow_copies)
#         input_grads = torch.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + ctx.input_params,
#             output_grads,
#             allow_unused=True,
#         )
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + input_grads


# Fixes: When we set unet parameters with requires_grad=False, the original CheckpointFunction
# still tries to compute gradient for unet parameters.
# https://discuss.pytorch.org/t/get-runtimeerror-one-of-the-differentiated-tensors-does-not-require-grad-in-pytorch-lightning/179738/6
# class CheckpointFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
#                                    "dtype": torch.get_autocast_gpu_dtype(),
#                                    "cache_enabled": torch.is_autocast_cache_enabled()}
#         with torch.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with torch.enable_grad(), \
#                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             output_tensors = ctx.run_function(*shallow_copies)
#         grads = torch.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + [x for x in ctx.input_params if x.requires_grad],
#             output_grads,
#             allow_unused=True,
#         )
#         grads = list(grads)
#         # Assign gradients to the correct positions, matching None for those that do not require gradients
#         input_grads = []
#         for tensor in ctx.input_tensors + ctx.input_params:
#             if tensor.requires_grad:
#                 input_grads.append(grads.pop(0))  # Get the next computed gradient
#             else:
#                 input_grads.append(None)  # No gradient required for this tensor
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + tuple(input_grads)
class CheckpointFunction(torch.autograd.Function):          # 【融合RGB图像方法二】
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        # 记录 None 值的位置
        ctx.none_indices = [i for i, x in enumerate(args[:length]) if x is None]
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # 处理输入张量，将 None 保留
        ctx.input_tensors = [
            x.detach().requires_grad_(True) if x is not None else None
            for x in ctx.input_tensors
        ]

        # 过滤掉 None 值，以便进行梯度计算
        valid_input_tensors = [x for x in ctx.input_tensors if x is not None]
        valid_input_params = [x for x in ctx.input_params if x is not None and x.requires_grad]

        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 复制有效输入张量
            shallow_copies = [x.view_as(x) for x in valid_input_tensors]
            # 重新构建输入参数，确保顺序正确
            input_args = []
            tensor_index = 0
            for i in range(len(ctx.input_tensors)):
                if i in ctx.none_indices:
                    input_args.append(None)
                else:
                    input_args.append(shallow_copies[tensor_index])
                    tensor_index += 1
            output_tensors = ctx.run_function(*input_args)

        grads = torch.autograd.grad(
            output_tensors,
            valid_input_tensors + valid_input_params,
            output_grads,
            allow_unused=True,
        )

        grads = list(grads)
        input_grads = []
        tensor_index = 0
        # 插入 None 梯度以匹配原始输入数量
        for i in range(len(ctx.input_tensors)):
            if i in ctx.none_indices:
                input_grads.append(None)
            else:
                if ctx.input_tensors[i] is not None and ctx.input_tensors[i].requires_grad:
                    input_grads.append(grads.pop(0))
                else:
                    input_grads.append(None)

        # 处理参数梯度
        for tensor in ctx.input_params:
            if tensor is not None and tensor.requires_grad:
                input_grads.append(grads.pop(0))
            else:
                input_grads.append(None)

        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + tuple(input_grads)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)
    # # 动态选择能整除 channels 的最大 num_groups（不超过32）
    # num_groups = 32
    # while channels % num_groups != 0 and num_groups > 1:
    #     num_groups -= 1
    # print(num_groups, channels)
    # return GroupNorm32(num_groups, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
