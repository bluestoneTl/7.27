import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
# python train_stage2_vae.py --config configs/train/train_stage2.yaml --train_stage 1
'''
python train_stage2_vae.py --config configs/train/train_stage2.yaml \
--train_stage 2 \
--controlnet_lca_ckpt experiment/experiment_MGFA_2/stage2/checkpoints/0030000.pt  \
--ckpt_vae experiment/experiment_MGFA_2/stage2/checkpoints/0030000_vae.pt

'''


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )
        
    # 阶段特定的权重加载逻辑
    if args.train_stage == 1:
        # 阶段1：从UNet初始化LCA
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_lca_from_unet()
        if accelerator.is_main_process:
            print(
                f"Stage 1: LCA initialized from UNet\n"
                f"New zero weights: {init_with_new_zero}\n"
                f"Scratch weights: {init_with_scratch}"
            )
    else:
        # 阶段2：加载LCA权重，初始化RCA
        if args.controlnet_lca_ckpt:
            # cldm.load_controlnet_lca_from_ckpt(torch.load(args.controlnet_lca_ckpt))
            ckpt = torch.load(args.controlnet_lca_ckpt)
            cldm.load_controlnet_lca_from_ckpt(ckpt['controlnet_LCA'])
            if accelerator.is_main_process:
                print(f"Stage 2: LCA loaded from {args.controlnet_lca_ckpt}")
        else:
            raise ValueError("Stage 2 requires --controlnet_lca_ckpt argument")
        
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_rca_from_unet()
        if accelerator.is_main_process:
            print(
                f"Stage 2: RCA initialized from UNet\n"
                f"New zero weights: {init_with_new_zero}\n"
                f"Scratch weights: {init_with_scratch}"
            )
        
    # 阶段2：加载训练好的VAE权重
    if args.train_stage == 2 and hasattr(args, 'ckpt_vae') and args.ckpt_vae:
        try:
            vae_ckpt = torch.load(args.ckpt_vae, map_location="cpu")
            if 'state_dict' in vae_ckpt:
                vae_ckpt = vae_ckpt['state_dict']
            cldm.vae.load_state_dict(vae_ckpt)
            if accelerator.is_main_process:
                print(f"Stage 2: VAE weights loaded from {args.ckpt_vae}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Error loading VAE weights from {args.ckpt_vae}: {e}")

    # if cfg.train.resume:  
    #     cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
    #     if accelerator.is_main_process:
    #         print(
    #             f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
    #         )
    # else:
    #     init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
    #     if accelerator.is_main_process:
    #         print(
    #             f"strictly load controlnet weight from pretrained SD\n"
    #             f"weights initialized with newly added zeros: {init_with_new_zero}\n"
    #             f"weights initialized from scratch: {init_with_scratch}"
    #         )

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Setup optimizer:
    if args.train_stage == 1:
        parameters_to_optimize = list(cldm.controlnet_LCA.parameters()) + list(cldm.vae.parameters())
        print("Training stage 1: Optimizing LCA only")
    else:
        parameters_to_optimize = list(cldm.controlnet_LCA.parameters()) + list(cldm.controlnet_RCA.parameters()) + list(cldm.vae.parameters())
        print("Training stage 2: Optimizing LCA + RCA")
    opt = torch.optim.AdamW(parameters_to_optimize, lr=cfg.train.learning_rate)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    step_vae_loss = []  # 用于记录每个step的vae_loss
    epoch_vae_loss = []  # 用于记录每个epoch的vae_loss
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False, is_first_stage=(args.train_stage == 1)
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt, rgb = batch
            # gt shape: torch.Size([16, 512, 512, 3])
            # lq shape: torch.Size([16, 512, 512, 3])
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            rgb = rearrange(rgb, "b h w c -> b c h w").contiguous().float()
            # gt shape: torch.Size([16, 3, 512, 512])
            # lq shape: torch.Size([16, 3, 512, 512])
            # rgb shape: 方法一是 torch.Size([16, 3, 512, 512])   方法二是 torch.Size([16, 1, 768])
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                cond = pure_cldm.prepare_condition(clean, prompt, rgb)      # 【融合RGB图像方法二】
                # cond shape: torch.Size([16, 4, 64, 64])
                #  对RGB执行操作  和 cond 进行concat  卷积下降 作为新的condtition   
                # noise augmentation
                cond_aug = copy.deepcopy(cond)      # 增强？
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug, is_first_stage=(args.train_stage == 1))       # 这里用cldm模型，计算损失，后续关键步骤的入口

            # 将解码器的输出与原始输入比较。目的是更新vae的参数
            lq = lq * 2 - 1     # 
            z_lq = pure_cldm.vae_encode(lq)
            reconstructed = pure_cldm.vae_decode(z_lq)         
            vae_loss = torch.nn.functional.mse_loss(reconstructed, lq)   
            loss = loss + vae_loss * 0.1        # 调整比例

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            step_vae_loss.append(vae_loss.item())  # 记录当前step的vae_loss
            epoch_vae_loss.append(vae_loss.item())  # 记录epoch的vae_loss
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}, VAE Loss: {vae_loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_vae_loss = (
                    accelerator.gather(
                        torch.tensor(step_vae_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                step_vae_loss.clear()  # 清空当前记录的vae_loss
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)
                    writer.add_scalar("loss/vae_loss_simple_step", avg_vae_loss, global_step)  # 记录平均vae_loss到TensorBoard

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # checkpoint = pure_cldm.controlnet.state_dict()
                    checkpoint = {
                        'controlnet_LCA': pure_cldm.controlnet_LCA.state_dict(),
                        'controlnet_RCA': pure_cldm.controlnet_RCA.state_dict()
                    }
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

                    # 保存vae
                    checkpoint_vae = pure_cldm.vae.state_dict()        
                    ckpt_path_vae = f"{ckpt_dir}/{global_step:07d}_vae.pt"
                    torch.save(checkpoint_vae, ckpt_path_vae)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_clean = clean[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]

                # 查看【vae_test】的结果
                log_reconstructed = reconstructed[:N]       # gt 解码后的结果

                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            # ("image/lq", log_lq),
                            ("image/lq",(log_lq + 1) / 2),  
                            ("image/condition", log_clean),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/condition_aug_decoded",
                                (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                            (
                                "image/reconstructed_lq",
                                (log_reconstructed + 1) / 2,        # 
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        avg_epoch_vae_loss = (
            accelerator.gather(torch.tensor(epoch_vae_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        epoch_vae_loss.clear()  # 清空epoch记录的vae_loss
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)
            writer.add_scalar("loss/vae_loss_simple_epoch", avg_epoch_vae_loss, global_step)  # 记录epoch的平均vae_loss

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_stage", type=int, default=1)
    parser.add_argument("--controlnet_lca_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_vae", type=str, default=None)
    args = parser.parse_args()
    main(args)
