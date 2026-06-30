import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MLP(nn.Module):
    def __init__(self, channels: list, enc: bool, has_ln=True):
        super().__init__()

        layers = []
        for i in range(1, len(channels) - 1):
            layers.append(nn.Linear(channels[i - 1], channels[i]))
            if has_ln:
                layers.append(nn.LayerNorm(channels[i], eps=1e-6))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(channels[-2], channels[-1]))
        if has_ln and enc:
            layers.append(nn.LayerNorm(channels[-1], eps=1e-6))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class CrossAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y) -> torch.Tensor:
        B, N, C = y.shape
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = q.unbind(0)[0]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)

        x = self.linear(x)
        return x

class TrajDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.norm0 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_ca, scale_ca, gate_ca = self.adaLN_modulation(t).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class TrajDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        hidden_size=256,
        frame_num=8,
        depth=2,
        num_heads=8,
        mlp_ratio=2.0,
    ):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.frame_num = frame_num
        
        self.x_embedder =  MLP(
                channels=[self.in_channels, hidden_size],
                enc=True,
            )
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.temp_embed = nn.Parameter(torch.zeros(1, frame_num, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TrajDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size, self.frame_num)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_trajmap_coord(self, bev_shape=(128, 256), grid_size=0.25):
        # real-world range
        x_min, x_max = 0, bev_shape[0] * grid_size
        y_min, y_max = -(bev_shape[1]//2 * grid_size), bev_shape[1]//2 * grid_size
        coord_x = torch.arange(x_min, x_max, grid_size) + (grid_size*0.5)
        coord_y = torch.arange(y_min, y_max, grid_size) + (grid_size*0.5)
        coord = torch.stack(torch.meshgrid(coord_x, coord_y), dim=-1) 
        return coord # torch.tensor: h, w, 2

    def forward(self, x, t, ego_query):
        """
        params:
        x: (N, T, 2) noised traj points
        t: (N, ) diffusion timestamps
        ego_query: (N, 4096)
        """

        
        x = self.x_embedder(x) # N T C
        x = x + self.temp_embed # N T C
        t = self.t_embedder(t) + ego_query.squeeze()
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)

        return x

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(length, dtype=np.float32)  # [..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):  # hidden_size=embed_dim 1152 pos=256
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)

    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    # print("omega",omega.shape) #288
    # print("pos",pos.shape)

    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb  #  256 576