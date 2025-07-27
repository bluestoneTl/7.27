from argparse import ArgumentParser, Namespace

import torch

from accelerate.utils import set_seed
from diffbir.inference import (
    BSRInferenceLoop,
    BFRInferenceLoop,
    BIDInferenceLoop,
    UnAlignedBFRInferenceLoop,
    CustomInferenceLoop,
)


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        if device == "mps":
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model parameters
    parser.add_argument(    # 要执行的任务。如果使用的是自训练模型，则忽略此选项
        "--task",
        type=str,
        default="sr",
        choices=["sr", "face", "denoise", "unaligned_face"],
        help="Task you want to do. Ignore this option if you are using self-trained model.",
    )
    parser.add_argument(    # 输出的放大因子
        "--upscale", type=float, default=4, help="Upscale factor of output."
    )
    parser.add_argument(    # DiffBIR模型版本
        "--version",
        type=str,
        default="v2.1",
        choices=["v1", "v2", "v2.1", "custom"],
        help="DiffBIR model version.",
    )
    parser.add_argument(    # 训练配置文件的路径。仅当版本为'custom'时生效
        "--train_cfg",
        type=str,
        default="",
        help="Path to training config. Only works when version is custom.",
    )
    parser.add_argument(    # 要测试的cldm模型的路径。仅当版本为'custom'时生效
        "--ckpt",
        type=str,
        default="",
        help="Path to saved checkpoint. Only works when version is custom.",
    )
    # sampling parameters
    parser.add_argument(    # 采样器类型，同的采样器可能会产生非常不同的样本
        "--sampler",
        type=str,
        default="edm_dpm++_3m_sde",
        choices=[
            "dpm++_m2",
            "spaced",
            "ddim",
            "edm_euler",
            "edm_euler_a",
            "edm_heun",
            "edm_dpm_2",
            "edm_dpm_2_a",
            "edm_lms",
            "edm_dpm++_2s_a",
            "edm_dpm++_sde",
            "edm_dpm++_2m",
            "edm_dpm++_2m_sde",
            "edm_dpm++_3m_sde",
        ],
        help="Sampler type. Different samplers may produce very different samples.",
    )
    parser.add_argument(    # 采样步数，越多步数，细节越多
        "--steps",
        type=int,
        default=10,
        help="Sampling steps. More steps, more details.",
    )
    parser.add_argument(    # 对于DiffBIR v1和v2，将起始点类型设置为'cond'可以使结果更加稳定，并确保来自ODE采样器（如DDIM和DPMS）的结果正常。但是，此调整可能会导致样本质量下降。
                            # 'noise'：以噪声作为起始点 'cond'：以条件作为起始点
        "--start_point_type",
        type=str,
        choices=["noise", "cond"],
        default="noise",
        help=(
            "For DiffBIR v1 and v2, setting the start point types to 'cond' can make the results much more stable "
            "and ensure that the outcomes from ODE samplers like DDIM and DPMS are normal. "
            "However, this adjustment may lead to a decrease in sample quality."
        ),  
    )
    parser.add_argument(    # 为阶段1模型启用分块推理，以减少GPU内存使用
        "--cleaner_tiled",
        action="store_true",
        help="Enable tiled inference for stage-1 model, which reduces the GPU memory usage.",
    )
    parser.add_argument(    # 阶段1模型分块推理的每个分块的大小
        "--cleaner_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(    # 阶段1模型分块推理的每个分块之间的步长
        "--cleaner_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(    # 为AE编码器启用分块推理，以减少GPU内存使用
        "--vae_encoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE encoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(    # AE编码器分块推理的每个分块的大小
        "--vae_encoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(    # 为AE解码器启用分块推理，以减少GPU内存使用
        "--vae_decoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE decoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(    # AE解码器分块推理的每个分块的大小
        "--vae_decoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(    # cldm模型启用分块采样，以减少GPU内存使用
        "--cldm_tiled",
        action="store_true",
        help="Enable tiled sampling, which reduces the GPU memory usage.",
    )
    parser.add_argument(    # cldm模型分块采样的每个分块的大小
        "--cldm_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(    # cldm模型分块采样的每个分块之间的步长
        "--cldm_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(    # 要使用的captioner模型
        "--captioner",
        type=str,
        choices=["none", "llava", "ram"],
        default="llava",
        help="Select a model to describe the content of your input image.",
    )
    parser.add_argument(    # 正向提示词
        "--pos_prompt",
        type=str,
        default=DEFAULT_POS_PROMPT,
        help=(
            "Descriptive words for 'good image quality'. "
            "It can also describe the things you WANT to appear in the image."
        ),
    )
    parser.add_argument(    # 负向提示词
        "--neg_prompt",
        type=str,
        default=DEFAULT_NEG_PROMPT,
        help=(
            "Descriptive words for 'bad image quality'. "
            "It can also describe the things you DON'T WANT to appear in the image."
        ),
    )
    parser.add_argument(    # 无分类器引导尺度
        "--cfg_scale", type=float, default=6.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(    # 将cfg尺度从1逐渐增加到('cfg_scale' + 1)
        "--rescale_cfg",
        action="store_true",
        help="Gradually increase cfg scale from 1 to ('cfg_scale' + 1)",
    )
    parser.add_argument(    # 噪声增强的级别。噪声越多，创意越多
        "--noise_aug",
        type=int,
        default=0,
        help="Level of noise augmentation. More noise, more creative.",
    )
    parser.add_argument(    # 采样中的随机性。仅适用于某些edm采样器
        "--s_churn",
        type=float,
        default=0,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(    # 为采样添加随机性的最小sigma值。仅适用于某些edm采样器
        "--s_tmin",
        type=float,
        default=0,
        help="Minimum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(    # 采样中的最大sigma值。仅适用于某些edm采样器
        "--s_tmax",
        type=float,
        default=300,
        help="Maximum  sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(    # 采样中的噪声。仅适用于某些edm采样器
        "--s_noise",
        type=float,
        default=1,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(    # ？？？？啊，不是哥们
        "--eta",
        type=float,
        default=1,
        help="I don't understand this parameter. Leave it as default.",
    )
    parser.add_argument(    # 求解器的阶数。仅适用于edm_lms采样器
        "--order",
        type=int,
        default=1,
        help="Order of solver. Only works with edm_lms sampler.",
    )
    parser.add_argument(    # 来自ControlNet的控制强度。强度越小，创意越多
        "--strength",
        type=float,
        default=1,
        help="Control strength from ControlNet. Less strength, more creative.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Nothing to say.")
    # guidance parameters
    parser.add_argument(    # 启用引导
        "--guidance", action="store_true", help="Enable restoration guidance."
    )
    parser.add_argument(    # 恢复引导的损失函数  mse：均方误差  w_mse：加权均方误差
        "--g_loss",
        type=str,
        default="w_mse",
        choices=["mse", "w_mse"],
        help="Loss function of restoration guidance.",
    )
    parser.add_argument(    # 优化引导损失函数的学习率
        "--g_scale",
        type=float,
        default=0.0,
        help="Learning rate of optimizing the guidance loss function.",
    )
    # common parameters
    parser.add_argument(    # 输入图像的路径
        "--input",
        type=str,
        required=True,
        help="Path to folder that contains your low-quality images.",
    )
    parser.add_argument(    # 输入rgb图像的路径
        "--edge_path",
        type=str,
        required=True,
        help="Path to folder that contains your low-quality images.",
    )
    parser.add_argument(    # 每个图像的样本数量
        "--n_samples", type=int, default=1, help="Number of samples for each image."
    )
    parser.add_argument(    # 输出的路径
        "--output", type=str, required=True, help="Path to save restored results."
    )
    parser.add_argument("--seed", type=int, default=231)
    # mps has not been tested
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(    # 
        "--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--llava_bit", type=str, default="4", choices=["16", "8", "4"])

    parser.add_argument('--condition_path', type=str, help='Path to the feature directory')

    parser.add_argument('--train_stage', type=int, help='Path to the feature directory')

    return parser.parse_args()


def main():
    args = parse_args()
    args.device = check_device(args.device)
    set_seed(args.seed)

    if args.version != "custom":
        loops = {
            "sr": BSRInferenceLoop,
            "denoise": BIDInferenceLoop,
            "face": BFRInferenceLoop,
            "unaligned_face": UnAlignedBFRInferenceLoop,
        }
        loops[args.task](args).run()
    else:
        CustomInferenceLoop(args).run()
    print("done!")


if __name__ == "__main__":
    main()
