from typing import Tuple, Set, List, Dict

import torch
from torch import nn
import torch as th
from .controlnet import ControlledUnetModel, ControlNet, ControlNet_LCA, ControlNet_RCA
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook
from .util import conv_nd, linear, zero_module, timestep_embedding, exists
import torch.nn.functional as F
def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, controlnet_RCA_cfg, latent_scale_factor, use_fp16=False,
    ):
        super().__init__()
        self.controlnet_LCA = ControlNet_LCA(**controlnet_cfg)
        # self.controlnet_RCA = ControlNet_RCA(**controlnet_RCA_cfg)
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13
        self.dtype = th.float16 if use_fp16 else th.float32

        self.z_ref_proj = nn.Conv2d(4, 320, kernel_size=1)  # 将z_ref的通道数从 4 变为 320
        self.adjust_conv_1 = nn.Conv2d(640, 320, kernel_size=1)
        self.adjust_conv_2 = nn.Conv2d(1280, 640, kernel_size=1)
        # self.adjust_conv_3 = nn.Conv2d(1280, 1280, kernel_size=1)

        self.conv = nn.Conv2d(
            in_channels=6,  # 输入通道数，因为输入张量的通道数为 6
            out_channels=3,  # 输出通道数，将通道数从 6 降低到 3
            kernel_size=1,  # 使用 1x1 卷积核，不改变特征图的大小
            stride=1,  # 步长为 1，保持尺寸不变
            padding=0  # 不使用填充，因为 1x1 卷积核且步长为 1 时不需要填充
        )
    
    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
        # for module in [self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:       # 【test】
        # self.controlnet.load_state_dict(sd, strict=True)
        self.controlnet_LCA.load_state_dict(sd["controlnet_LCA"], strict=True)
        self.controlnet_RCA.load_state_dict(sd["controlnet_RCA"], strict=True)

    @torch.no_grad()
    def load_controlnet_lca_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet_LCA.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    @torch.no_grad()
    def load_controlnet_lca_from_unet(self) -> Tuple[Set[str]]:
        # 初始化LCA模块
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet_LCA.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet_LCA.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    @torch.no_grad()
    def load_controlnet_rca_from_unet(self) -> Tuple[Set[str]]:
        # 仅初始化RCA模块
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet_RCA.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet_RCA.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        # cond_RGB: torch.Tensor,  # 新增条件RGB特征作为输入
        cond_edge: torch.Tensor,  # 新增条件边缘特征作为输入
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),    # prompt
            # cond_img shape: torch.Size([16, 3, 512, 512])    # 输入的cond_img的形状
            c_img=self.vae_encode(
                cond_img * 2 - 1,   # 像素值范围从 [0, 1] 转换为 [-1, 1]
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ), 
            # c_rgb=cond_RGB,               # 【融合RGB图像方法二】               
            # c_rgb = self.vae_encode(
            #     cond_RGB * 2 - 1,   # 像素值范围从 [0, 1] 转换为 [-1, 1]
            #     sample=False,
            #     tiled=tiled,
            #     tile_size=tile_size,
            # ),
            c_edge = self.vae_encode(
                cond_edge * 2 - 1,   # 像素值范围从 [0, 1] 转换为 [-1, 1]
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),            
            # c_img shape: torch.Size([16, 4, 64, 64])   # 原本输出的结果
        )


    def forward(self, x_noisy, t, cond, is_first_stage):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        # c_rgb = cond["c_rgb"]               # rgb图像 vae编码后的特征
        c_edge = cond["c_edge"]             # 边缘图像 vae编码后的特征
        
        # control = self.controlnet(x=x_noisy, hint=c_img, rgb=c_rgb, timesteps=t, context=c_txt)  # 【融合RGB图像方法二】
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        # eps = self.unet(
        #     x=x_noisy,
        #     timesteps=t,
        #     context=c_txt,
        #     # control=control,
        #     hint=c_img,
        #     rgb=c_rgb,
        #     only_mid_control=False,
        #     is_first_stage=is_first_stage,
        # )

        # 1. 提取Unet解码器特征
        t_emb = timestep_embedding(t, self.unet.model_channels, repeat_only=False)
        emb = self.unet.time_embed(t_emb)
        h, emb, context = map(lambda t: t.type(self.dtype), (x_noisy, emb, c_txt))
        hs = []
        for module in self.unet.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        # 2. 计算控制信号

        # 只训练LCA部分
        z_hint = self.controlnet_LCA.resblock(c_img) 
        if c_edge.shape[2] != 64:
            c_edge = F.interpolate(c_edge, size=64, mode='bilinear', align_corners=True)
        control = self.controlnet_LCA(x=x_noisy, hint=z_hint, edge=c_edge, timesteps=t, context=c_txt, unet_encoder_results=hs)  # LCA加入RGB图像，用交叉注意力融合

        # if is_first_stage:
        #     # z_hint = self.controlnet_LCA.RLF(gt=c_img, z_lq=c_rgb)           # lq也加RLF，与RGB图像融合
        #     z_hint = c_img
        #     control = self.controlnet_LCA(x=x_noisy, hint=z_hint, timesteps=t, context=c_txt, unet_encoder_results=hs)
        # else:
        #     # z_ref = self.controlnet_RCA.RLF(gt=c_rgb, z_lq=c_img)         # 实验，只有lq图像加RLF，与RGB图像融合
        #     z_ref = c_rgb
        #     z_hint = self.controlnet_LCA.RLF(gt=c_img, z_lq=c_rgb)
        #     control = self.control_DCA(x=x_noisy, hint=z_hint, z_ref=z_ref, timesteps=t, context=c_txt, unet_encoder_results=hs)

        control = [c * scale for c, scale in zip(control, self.control_scales)]

        # 3. 应用控制信号
        h = self.unet.middle_block(h, emb, context)
        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.unet.output_blocks):
            if control is not None:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, c_txt)

        # 4. 输出结果
        h = h.type(x_noisy.dtype)
        eps = self.unet.out(h)
        return eps
    
    def odd_even_channel_cat(self, z_ref, c_edge):
        # 融合z_ref和边缘图    
        batch_size, channels, height, width = z_ref.shape
        result_channels = channels * 2
        result = torch.zeros((batch_size, result_channels, height, width), dtype=z_ref.dtype)
        result[:, 0::2] = z_ref
        result[:, 1::2] = c_edge
        return result.to(z_ref.device)

    def adjust_conv(self, x, i):
        if i == 4: 
            return self.adjust_conv_1(x).to(x.device)
        elif i == 7:
            return self.adjust_conv_2(x).to(x.device)
    
    def control_DCA(self, x, hint, z_ref, timesteps, context, unet_encoder_results):
            t_emb = timestep_embedding(timesteps, self.unet.model_channels, repeat_only=False)
            emb = self.unet.time_embed(t_emb)

            # x = x + hint
            x = torch.cat((x, hint), dim=1)

            outs = []
            lca_features = []
            rca_features = []

            z_ref = self.z_ref_proj(z_ref)

            h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))    
            for i, (module_LCA, zero_conv_LCA, module_RCA, zero_conv_RCA) in enumerate(
                    zip(self.controlnet_LCA.input_blocks, self.controlnet_LCA.zero_convs,
                        self.controlnet_RCA.input_blocks, self.controlnet_RCA.zero_convs)):
                # 在 LCA 中计算
                h_lca = module_LCA(h, emb, context)     
                lca_features.append(zero_conv_LCA(h_lca, emb, context))

                # h shape: torch.Size([16, 8, 64, 64])
                # lca_features[i] shape: torch.Size([16, 320, 64, 64])
                # z_ref shape: torch.Size([16, 320, 64, 64])       通道数扩充到了320

                # 在 RCA 中计算，并合并到 h
                if i == 0:
                    h_rca = module_RCA(lca_features[i] + z_ref, emb, context)      
                else :
                    if i == 3 or i == 6 or i == 9:  
                        lca_feature = F.interpolate(lca_features[i], size=rca_features[i-1].shape[2:], mode='bilinear', align_corners=True)
                    elif i == 4 or i == 7 : 
                        lca_feature = self.adjust_conv(lca_features[i],i) 
                    else:
                        lca_feature = lca_features[i]
                    
                    h_rca = module_RCA(lca_feature + rca_features[i-1], emb, context)

                rca_features.append(zero_conv_RCA(h_rca, emb, context))

                h = lca_features[i] + rca_features[i]
                h = h + unet_encoder_results[i]  # 融合编码器部分的中间结果

                outs.append(h)

            # 处理中间块
            h_rca_middle = self.controlnet_RCA.middle_block(rca_features[-1], emb, context)
            h_lca_middle = self.controlnet_LCA.middle_block(h[-1] + h_rca_middle, emb, context)
            
            h_middle = h_lca_middle 
            outs.append(h_middle)

            return outs
    
    # def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
    #     self.unet.dtype = dtype
    #     self.controlnet.dtype = dtype
    #     # convert unet blocks to dtype
    #     for module in [
    #         self.unet.input_blocks,
    #         self.unet.middle_block,
    #         self.unet.output_blocks,
    #     ]:
    #         module.type(dtype)
    #     # convert controlnet blocks and zero-convs to dtype
    #     for module in [
    #         self.controlnet.input_blocks,
    #         self.controlnet.zero_convs,
    #         self.controlnet.middle_block,
    #         self.controlnet.middle_block_out,
    #     ]:
    #         module.type(dtype)

    #     def cast_groupnorm_32(m):
    #         if isinstance(m, GroupNorm32):
    #             m.type(torch.float32)

    #     # GroupNorm32 only works with float32
    #     for module in [
    #         self.unet.input_blocks,
    #         self.unet.middle_block,
    #         self.unet.output_blocks,
    #     ]:
    #         module.apply(cast_groupnorm_32)
    #     for module in [
    #         self.controlnet.input_blocks,
    #         self.controlnet.zero_convs,
    #         self.controlnet.middle_block,
    #         self.controlnet.middle_block_out,
    #     ]:
    #         module.apply(cast_groupnorm_32)

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        # Convert main Unet and ControlNets to target dtype
        self.unet.dtype = dtype
        self.controlnet_LCA.dtype = dtype
        # self.controlnet_RCA.dtype = dtype

        # Convert Unet blocks
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)

        # Convert ControlNet_LCA blocks
        for module in [
            self.controlnet_LCA.input_blocks,
            self.controlnet_LCA.zero_convs,
            self.controlnet_LCA.middle_block,
            self.controlnet_LCA.middle_block_out,
        ]:
            module.type(dtype)

        # # Convert ControlNet_RCA blocks
        # for module in [
        #     self.controlnet_RCA.input_blocks,
        #     self.controlnet_RCA.zero_convs,
        #     self.controlnet_RCA.middle_block,
        #     self.controlnet_RCA.middle_block_out,
        # ]:
        #     module.type(dtype)

        # Convert new custom layers
        self.z_ref_proj.type(dtype)
        self.adjust_conv_1.type(dtype)
        self.adjust_conv_2.type(dtype)
        self.conv.type(dtype)

        # GroupNorm32 only works with float32
        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # Apply to Unet
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)

        # Apply to ControlNet_LCA
        for module in [
            self.controlnet_LCA.input_blocks,
            self.controlnet_LCA.zero_convs,
            self.controlnet_LCA.middle_block,
            self.controlnet_LCA.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)

        # # Apply to ControlNet_RCA
        # for module in [
        #     self.controlnet_RCA.input_blocks,
        #     self.controlnet_RCA.zero_convs,
        #     self.controlnet_RCA.middle_block,
        #     self.controlnet_RCA.middle_block_out,
        # ]:
        #     module.apply(cast_groupnorm_32)

        return self