from packaging import version
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from .util import checkpoint, zero_module, exists, default
from .config import Config, AttnMode


# CrossAttn precision handling
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
        # query_dim=320, context_dim=1024, heads=5, dim_head=64, dropout=0.0
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0): 
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (vanilla). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)             # 
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            # with torch.autocast(enabled=False, device_type = 'cuda'):
            with torch.autocast(
                enabled=False,
                device_type="cuda" if str(x.device).startswith("cuda") else "cpu",
            ):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (xformers). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class SDPCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (sdp). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = F.scaled_dot_product_attention(q, k, v)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        AttnMode.VANILLA: CrossAttention,  # vanilla attention
        AttnMode.XFORMERS: MemoryEfficientCrossAttention,
        AttnMode.SDP: SDPCrossAttention,
    }

    def __init__(
        self,
        dim,    # 320
        n_heads,     # 5
        d_head,      # 64
        dropout=0.0,
        context_dim=None,    # 1024/768
        rgb_dim=None,  # 新增参数      【融合RGB图像方法二】
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[Config.attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,    # 
            context_dim=context_dim,    # 1024/768
            heads=n_heads,       # 5
            dim_head=d_head,      # 64
            dropout=dropout,
        )  # is self-attn if context is none
        self.attn_rgb1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=rgb_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.attn_rgb2 = attn_cls(  # 新增模块，用于处理rgb特征的交叉注意力,     【融合RGB图像方法二】
            query_dim=dim,
            context_dim=rgb_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm1_rgb = nn.LayerNorm(dim)
        self.norm2_rgb = nn.LayerNorm(dim)  
        self.checkpoint = checkpoint

    def forward(self, x, context=None, rgb=None):
        return checkpoint(
            self._forward, (x, context, rgb), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None, rgb=None):
        # x = (
        #     self.attn1(
        #         self.norm1(x), context=context if self.disable_self_attn else None
        #     )
        #     + x
        # )
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x

        # 【注意力写法2】  注意参数不能共享
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        if rgb is not None:
            x = (
                self.attn_rgb1(
                    self.norm1_rgb(x), context=rgb if self.disable_self_attn else None
                )
                + x
            )
            
        x = self.attn2(self.norm2(x), context=context) + x
        if rgb is not None:
            x = self.attn_rgb2(self.norm2_rgb(x), context=rgb) + x

        x = self.ff(self.norm3(x)) + x

        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,    # 5
        d_head,     # 64
        depth=1,
        dropout=0.0,
        context_dim=None,       # 1024
        rgb_dim=768,  # 新增参数      【融合RGB图像方法二】
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if exists(rgb_dim) and not isinstance(rgb_dim, list):       # 【融合RGB图像方法二】
            rgb_dim = [rgb_dim]        
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,    
                    n_heads,     
                    d_head,      
                    dropout=dropout,
                    context_dim=context_dim[d],     # 1024
                    rgb_dim=rgb_dim[d],  # 新增参数      【融合RGB图像方法二】
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, edge=None):  ## 16 1 512   16 4 512 512     # 在这里将clip提取的特征与x使用 ，用交叉注意力
    # def forward(self, x, context=None):  
        # note: if no context is given, cross-attention defaults to self-attention
        # x shape:　torch.Size([16, 320, 64, 64])
        # context shape: torch.Size([16, 77, 1024])                                                                          
        # rgb shape: torch.Size([16, 4, 64, 64])                                                                                       
        # rgb shape: torch.Size([16, 4, 4096])     
        if edge is not None:
            edge = rearrange(edge, "b c h w -> b c (h w)").contiguous()
        if not isinstance(context, list):
            context = [context]     
        if not isinstance(edge, list):
            edge = [edge]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)       
        # 这里可以加判断处理，如果rgb是四维的，按和上面一样的操作 处理成三维，然后过self.proj_in卷积处理下（不行，得具体看看这个卷积是干什么的）

        for i, block in enumerate(self.transformer_blocks):
            # x = block(x, context=context[i])
            x = block(x, context=context[i], rgb=edge[i])

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        if not self.use_linear:
            x = self.proj_out(x)
        x = x + x_in      
    
        return x 
