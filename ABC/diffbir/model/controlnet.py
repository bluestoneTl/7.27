import torch
import torch as th
import torch.nn as nn

from .util import conv_nd, linear, zero_module, timestep_embedding, exists
from .attention import SpatialTransformer
from .unet import (
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
    UNetModel,
)

class ControlledUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_scales = [1.0] * 13

    def forward(                # x_noisy, t, c_txt, c_img, c_rgb
        self,
        x,
        timesteps=None,
        context=None,
        # control=None,
        hint=None,
        rgb=None,               # vae处理过的rgb图像
        only_mid_control=False,
        is_first_stage=True,    # 是否是一阶段训练（只训练LCA模块）
        **kwargs,
    ):
        hs = []         # UNet编码器部分的中间结果

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        if is_first_stage:
            # 用LCA计算控制信息,第一阶段
            control = self.controlnet_LCA(x=x, hint=hint, timesteps=timesteps, context=context, unet_encoder_results=hs)    
        else:
            # 用 LCA 和 RCA 计算控制信息，第二阶段
            z_ref = self.controlnet_RCA.RLF(gt_image=hint, z_lq=rgb, emb=emb) # rgb
            control = self.control_DCA(x=x, hint=hint, z_ref=z_ref, rgb=rgb, timesteps=timesteps, context=context, unet_encoder_results=hs)

        control = [c * scale for c, scale in zip(control, self.control_scales)]

        h = self.middle_block(h, emb, context)  
        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)
    
    def control_DCA(self, x, hint, z_ref, rgb, timesteps, context, unet_encoder_results):
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            x = x + hint

            outs = []
            lca_features = []
            rca_features = []

            h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))      
            for i, (module_LCA, zero_conv_LCA, module_RCA, zero_conv_RCA) in enumerate(
                    zip(self.controlnet_LCA.input_blocks, self.controlnet_LCA.zero_convs,
                        self.controlnet_RCA.input_blocks, self.controlnet_RCA.zero_convs)):
                # 在 LCA 中计算
                h_lca = module_LCA(h, emb, context)
                lca_features.append(zero_conv_LCA(h_lca))

                # 在 RCA 中计算，并合并到 h
                if i == 0:
                    h_rca = module_RCA(lca_features[i] + z_ref, emb, context, rgb)      
                else :
                    h_rca = module_RCA(lca_features[i] + rca_features[i-1], emb, context, rgb)
                rca_features.append(zero_conv_RCA(h_rca))

                h = lca_features[i] + rca_features[i]
                h = h + unet_encoder_results[i]  # 融合编码器部分的中间结果

                outs.append(h)

            # 处理中间块
            h_rca_middle = self.controlnet_RCA.middle_block(rca_features[-1], emb, context, rgb)
            h_lca_middle = self.controlnet_LCA.middle_block(h[-1] + h_rca_middle, emb, context)
            
            h_middle = h_lca_middle 
            outs.append(h_middle)

            return outs


class ControlNet(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        rgb_dim=None,  # 新增参数      【融合RGB图像方法二】
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"
            
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):         # num_res_blocks=2
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]  # num_res_blocks=[2,2,2,2]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        dims, in_channels + hint_channels, model_channels, 3, padding=1     # cat就行   [2,8,320,3]
                    )
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):             # channel_mult=[1,2,4,4]
            for nr in range(self.num_res_blocks[level]):        # num_res_blocks=[2,2,2,2]
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:                 # attention_resolutions=[ 4, 2, 1 ]
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:                                  # legacy=false
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):         # disable_self_attentions=none
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer          # use_spatial_transformer: True
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                rgb_dim=rgb_dim,  # 新增参数      【融合RGB图像方法二】
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown      # resblock_updown: false
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer          # use_spatial_transformer: True
                else SpatialTransformer(  # always uses a self-attn     # 只有这里用到了context！ 看后续怎么融合
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    rgb_dim=rgb_dim,  # 新增参数      【融合RGB图像方法二】
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, rgb, timesteps, context,  **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        x = torch.cat((x, hint), dim=1)    # 【融合RGB图像方法二】 通道数 8     记得改配置文件！！  

        outs = []
        
        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        for i,(module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            h = module(h, emb, context, rgb)            
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context, rgb)      
        outs.append(self.middle_block_out(h, emb, context))

        return outs

# DCA中的LCA
class ControlNet_LCA(ControlNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.RLF = RLFModule(in_channels=4) 
        self.resblock = ResBlock2(in_channels=4)
    
    def forward(self, x, hint, edge, timesteps, context, unet_encoder_results, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        x = torch.cat((x, hint), dim=1)    # 【融合RGB图像方法二】 通道数 8     记得改配置文件！！  保持cat就行

        outs = []
        
        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        for i,(module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            h = module(h, emb, context, edge)            
            h = h + unet_encoder_results[i]               # 在每层的调用中需要加入UNet编码器中间结果 
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context, edge)      
        outs.append(self.middle_block_out(h, emb, context))

        return outs    

class ResBlock2(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out
    
class RLFModule(nn.Module):
    def __init__(self, in_channels):
        super(RLFModule, self).__init__()
        self.resblock = ResBlock2(in_channels)
        self.z_lq_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, gt, z_lq):
        # gt经过残差块
        gt_output = self.resblock(gt)
        # z_lq经过卷积
        z_lq_output = self.z_lq_conv(z_lq)
        # 做加法
        output = gt_output + z_lq_output
        return output
    
class ControlNet_RCA(ControlNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #需要添加一个模块用于 处理gt,并与lq想加，即论文中的RLF模块  计算得到z_ref
        self.RLF = RLFModule(in_channels=4) 
        self.input_blocks
        


'''
    controlnet(LCA) 网络结构：
    self.time_embed = nn.Sequential(
        linear(model_channels, time_embed_dim),     model_channels=320, time_embed_dim=320 * 4 = 1280
        nn.SiLU(),
        linear(time_embed_dim, time_embed_dim),
    )

    self.input_blocks :
        初始卷积层：
        TimestepEmbedSequential(
            conv_nd(
                    dims, in_channels + hint_channels, model_channels, 3, padding=1     # [2,8,320,3]
            )
        )
        后续层，按 channel_mult=[1,2,4,4] 循环:

        ===第一层===
        ResBlock                    (ch=320,out_channels=320)
        SpatialTransformer          (ch=320)
        ResBlock                    (ch=320,out_channels=320)   
        SpatialTransformer          (ch=320)
        Downsample                  (ch=320,out_ch=320)

        ===第二层===
        ResBlock                    (ch=320,out_channels=640)
        SpatialTransformer          (ch=640)
        ResBlock                    (ch=640,out_channels=640)   
        SpatialTransformer          (ch=640)
        Downsample                  (ch=640,out_ch=640)

        ===第三层===
        ResBlock                    (ch=640,out_channels=1280)
        SpatialTransformer          (ch=1280)
        ResBlock                    (ch=1280,out_channels=1280)   
        SpatialTransformer          (ch=1280)
        Downsample                  (ch=1280,out_ch=1280)     

        ===第四层===
        ResBlock                    (ch=1280,out_channels=1280)
        ResBlock                    (ch=1280,out_channels=1280)      

    self.zero_convs       
        与 self.input_blocks 输入通道对应， [320,640,1280,1280]

    self.middle_block :
        ResBlock                    (ch=1280,out_channels=1280)
        SpatialTransformer          (ch=1280)
        ResBlock                    (ch=1280,out_channels=1280)   
    
    self.middle_block_out :
        zero_conv                   [1280]

'''

'''
    controlnet(RCA) 网络结构：
    self.time_embed = nn.Sequential(
        linear(model_channels, time_embed_dim),     model_channels=320, time_embed_dim=320 * 4 = 1280
        nn.SiLU(),
        linear(time_embed_dim, time_embed_dim),
    )

    self.input_blocks :
        初始卷积层：
        TimestepEmbedSequential(
            conv_nd(
                    dims, in_channels + hint_channels, model_channels, 3, padding=1     # [2,320,320,3]
            )
        )
        后续层，按 channel_mult=[1,2,4,4] 循环:

        ===第一层===
        ResBlock                    (ch=320,out_channels=320)
        SpatialTransformer          (ch=320)
        ResBlock                    (ch=320,out_channels=320)   
        SpatialTransformer          (ch=320)
        Downsample                  (ch=320,out_ch=320)

        ===第二层===
        ResBlock                    (ch=320,out_channels=640)
        SpatialTransformer          (ch=640)
        ResBlock                    (ch=640,out_channels=640)   
        SpatialTransformer          (ch=640)
        Downsample                  (ch=640,out_ch=640)

        ===第三层===
        ResBlock                    (ch=640,out_channels=1280)
        SpatialTransformer          (ch=1280)
        ResBlock                    (ch=1280,out_channels=1280)   
        SpatialTransformer          (ch=1280)
        Downsample                  (ch=1280,out_ch=1280)     

        ===第四层===
        ResBlock                    (ch=1280,out_channels=1280)
        ResBlock                    (ch=1280,out_channels=1280)      

    self.zero_convs       
        与 self.input_blocks 输入通道对应， [320,640,1280,1280]

    self.middle_block :
        ResBlock                    (ch=1280,out_channels=1280)
        SpatialTransformer          (ch=1280)
        ResBlock                    (ch=1280,out_channels=1280)   
    
    self.middle_block_out :
        zero_conv                   [1280]    
'''