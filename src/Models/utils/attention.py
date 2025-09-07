import warnings
import math

from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)

import xformers.ops as xops


from . import build_dropout

class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        
        num_heads (int): Parallel attention heads.
        
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        
        dropout_layer (dict): The dropout_layer used
            when adding the shortcut.
            Default: {
                     'drop_type': 'Dropout',
                     'drop_prob': 0.,
                     'inplace': None
                     }
            
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer: dict = {
                     'drop_type': 'Dropout',
                     'drop_prob': 0.,
                     'inplace': None
                     },
                 batch_first=False,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            **dropout_layer) if dropout_layer else nn.Identity()
        
        
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        
        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
        
             
            
            
class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
            
    Shape:
        x: (N, L, C)
        hw_shape: (H, W)
    or
        x: (N, H, W, C)
        
        out: same as x
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 use_rel_pos: bool = False,
                 ):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                window_size is not None
            ), "window_size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_embed_dims))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_embed_dims))
        

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.rel_pos_h, std=0.02)
        trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, hw_shape=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
            
        if hw_shape is None:
            assert len(x.shape) == 4, "shape of x must be (batch_size, H, W, C)"
            B, H, W, _ = x.shape
            # qkv with shape (3, B, nHead, H * W, C)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            # q, k, v with shape (B * nHead, H * W, C)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            
            q = q * self.scale
            attn = (q * self.scale) @ k.transpose(-2, -1)

            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
            
        else:
            assert len(x.shape) == 3, "shape of x must be (batch_size, L, C)"
            B, N, C = x.shape
            H, W = hw_shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
            # make torchscript happy (cannot use tensor as tuple)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.use_rel_pos:
                q_Bnheads = q.reshape(B*self.num_heads, N, C // self.num_heads)
                attn_Bheads = attn.reshape(B*self.num_heads, N, N)
                attn_Bheads = add_decomposed_rel_pos(attn_Bheads, q_Bnheads, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                attn = attn_Bheads.reshape(B, self.num_heads, N, N)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            
        return x
    
    def get_attn_map(self, x):
        assert len(x.shape) == 4, "shape of x must be (batch_size, H, W, C)"
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        
        q = q * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        return attn

         
         
# =======================================================================
# ===================== Cross Atettion ==================================         
        
class BaseAttention(nn.Module):      
    def __init__(self,
            embed_dims: int,
            num_heads: int,
            downsample_rate: int = 1,
            qkv_bias: bool=False, 
            qk_scale=None,
            xattn: bool=False,
            
            rope=None,
            
            ):
        super().__init__()  
        
        
        
        
        self.embedding_dim = embed_dims
        self.internal_dim = embed_dims // downsample_rate
        self.num_heads = num_heads
        
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.xattn = xattn
        assert self.internal_dim % num_heads == 0, "num_heads must divide embed_dims."
        
        self.rope = rope
        

        self.q_proj = nn.Linear(embed_dims, self.internal_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, self.internal_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dims, self.internal_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.internal_dim, embed_dims)
        
        self.init_weights()
        
    def init_weights(self):
        trunc_normal_init(self.q_proj, std=.02)
        trunc_normal_init(self.k_proj, std=.02)
        trunc_normal_init(self.v_proj, std=.02)
        trunc_normal_init(self.out_proj, std=.02)
         
         
    def forward_features(
        self,
        q,
        k, 
        v
    ):
        # Input projections
        B, N, C = q.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        # (N, num_heads, L, c // num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        
        if self.rope is not None:
            # q_t = q[:, :, 1:, :]
            q = self.rope(q)
            q = q.type_as(v)
            # q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            # k_t = k[:, :, 1:, :]
            k = self.rope(k)
            k = k.type_as(v)
            # k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        # Attention
        if not self.xattn:
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn * self.scale
            attn = torch.softmax(attn, dim=-1)

            # Get output
            out = attn @ v
            out = self._recombine_heads(out)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            out = xops.memory_efficient_attention(q, k, v)
            out = out.reshape(B, N, -1)
            
        out = self.out_proj(out)
        return out

         
    def forward(self,
                q,
                k=None,
                v=None,
                ):
        
        out = self.forward_features(q, k, v)

        return out
    
    
    # functional
    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C
        
         
class CrossAttention(BaseAttention):
    def __init__(self, 
                 embed_dims, 
                 num_heads, 
                 xattn, 
                 rope):
        super().__init__(embed_dims, num_heads, xattn=xattn, rope=rope)
        
        
    def forward_features(
        self,
        q,
        k, 
        v
    ):
        # Input projections
        B, N, C = q.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        # (N, num_heads, L, c // num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        
        if self.rope is not None:
            # q_t = q
            q = self.rope(q)
            q = q.type_as(v)
            # q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

        # Attention
        if not self.xattn:
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn * self.scale
            attn = torch.softmax(attn, dim=-1)

            # Get output
            out = attn @ v
            out = self._recombine_heads(out)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            out = xops.memory_efficient_attention(q, k, v)
            out = out.reshape(B, N, -1)
            
        out = self.out_proj(out)
        return out
    
    
    def forward(self, q, kv):
        out = self.forward_features(q, kv, kv)
        
        return out
             
         
    
class SelfAttention(BaseAttention):
    def __init__(self, 
                 embed_dims, 
                 num_heads, 
                 qkv_bias=False,
                 xattn=False, 
                 rope=None):
        super().__init__(embed_dims, num_heads, qkv_bias=qkv_bias, xattn=xattn, rope=rope)
        
    def forward(self, q):
        out = self.forward_features(q, q, q)
        
        return out
            


# ================== functions =========================

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn





# ==================== deprecated ====================

class WindowAttention_old(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x









