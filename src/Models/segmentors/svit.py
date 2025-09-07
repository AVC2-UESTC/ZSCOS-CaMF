import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union, Optional, Dict
from torch import Tensor
from timm.models.vision_transformer import Mlp, _init_vit_weights, trunc_normal_, named_apply

from torch.nn.init import normal_

import logging
import math
from functools import partial

from timm.models.layers import to_2tuple

from torch.nn import MultiheadAttention, Dropout, Linear, LayerNorm, Identity

import torch.utils.checkpoint as cp

from ops.modules import MSDeformAttn

from ..utils import resize


_logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 drop_path=0., device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else Identity()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if [int(x) for x in torch.__version__.split('+')[0].split('.')] <= [int(x) for x in '1.13.0'.split('.')]:
            why_not_sparsity_fast_path = 'require python version >= 1.13'
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )

        x = src
        if self.norm_first:
            x = x + self.drop_path1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            x = x + self.drop_path2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop_path1(self._sa_block(x, src_mask, src_key_padding_mask)))
            x = self.norm2(x + self.drop_path2(self._ff_block(x)))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def left_align_tokens2(x: Tensor, mask: Tensor):
    """
        x: tensor of shape B, L, D
        mask: boolean tensor of shape B, L
    """

    l_aligned_mask, indexes = torch.sort(mask.int(), dim=1, descending=True, stable=True)
    l_aligned_mask = l_aligned_mask.bool()  # bool --> int (sort) --> bool, because CUDA does not sort boolean tensor
    l_aligned_x = x[torch.arange(x.shape[0], device=x.device).unsqueeze(1), indexes]

    return l_aligned_x, l_aligned_mask


def set_inference(module: nn.Module, name, value: bool):
    if hasattr(module, 'inference'):
        module.inference = value
        # print(f'set {name}.inference to {value} ')


class SelectiveModule(nn.Module):
    def __init__(self, channels, rd_channels, hidden_channels, drop=0.,
                 tau=1.,
                 version=0, inference=False):
        super(SelectiveModule, self).__init__()
        self.inference = inference
        self.norm = nn.LayerNorm(channels)
        self.tau = tau
        self.version = version  # version 0 includes CLS token when selecting, version 1 excludes CLS while selecting
        assert version in (0, 1)
        if self.version == 0 or self.version == 1:
            self.mlp = Mlp(channels, hidden_channels, 2, act_layer=nn.GELU, drop=drop)
        self.gate2 = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        B, L, C = x.shape
        if self.version == 1:
            x = x[:, 1:, :]
        x = self.norm(x)
        x = self.mlp(x)  # shape (B,L,1) or (B,L,2)
        scale = self.gate2(x)  #shape (B,L,1) or (B,L,2)
        if not self.inference:
            selector = F.gumbel_softmax(scale, tau=self.tau, hard=True)[:, :, 0:1]  # shape (B, L, 1)
        else:
            selector = torch.argmin(scale, dim=-1, keepdim=True)
        diff_selector = selector
        if self.version == 1:
            selector = torch.cat((torch.ones(B, 1, 1, device=selector.device), selector), dim=1).bool().squeeze(2)
        else:  # self.version = 0
            selector = selector.bool().squeeze(2)

        return selector, diff_selector


class nnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., init_values=None,
                 drop_path=0., act_layer='gelu', norm_layer=nn.LayerNorm,
                 ):
        super(nnBlock, self).__init__()
        _logger.info('Warning: argument qkv_bias is not used by this model.')
        assert attn_drop == drop, 'attn_drop and drop are the same in nn.TransformerEncoder'
        assert norm_layer == nn.LayerNorm or norm_layer.func == nn.LayerNorm, 'nn.TransformerEncoder only supports LayerNorm'
        assert qkv_bias is True, 'pytorch Transformer uses qkv_bias'
        self.TransformerEncoderLayer = TransformerEncoderLayer(
            dim, num_heads, int(mlp_ratio * dim), dropout=drop, activation=act_layer, batch_first=True, norm_first=True,
            drop_path=drop_path)

    def forward(self, x, src_key_padding_mask=None):
        return self.TransformerEncoderLayer(x, src_key_padding_mask=src_key_padding_mask)


class SelectiveVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, fc_norm=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer='gelu', select_loc=[7, 8, 9, 10, 11, 12],
                 select_model_id=[0, 0, 0, 1, 1, 1], ratio_loss=False, keep_ratio=None, visualize=False,
                 inherit_mask=False, version=0, last_version=None, statistics=True, ratio_per_sample=False):
        super(SelectiveVisionTransformer, self).__init__()
        self.global_pool = global_pool
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_prefix_tokens = 1 if global_pool == 'token' else 0
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            nnBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    init_values=None, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        assert len(select_loc) == len(select_model_id)
        self.num_heads = num_heads
        self.sl_loc = select_loc
        self.sl_model_id = select_model_id
        self.selective_modules = nn.ModuleList([])

        for i in range(len(set(select_model_id))):
            if i == len(set(select_model_id)) - 1 and last_version is not None:
                version = last_version
            self.selective_modules.append(SelectiveModule(
                embed_dim, embed_dim // 4, embed_dim // 4,
                version=version))
        self.ratio_loss = ratio_loss
        if keep_ratio is not None:
            assert len(keep_ratio) == len(set(self.sl_model_id))
            self.keep_ratio = keep_ratio
        self.visualize = visualize
        self.inherit_mask = inherit_mask
        self.normal_path = False
        self.fast_path = False
        self.statistics = statistics
        self.ratio_per_sample = ratio_per_sample
        self.grad_checkpointing = False

        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def set_all_inference_to(self, value: bool):
        named_apply(partial(set_inference, value=value), self, name=type(self).__name__)


    def forward(self, x):
        raise NotImplementedError  # implemented and called in adapter_modules.py




class SelectiveViTAdapter(SelectiveVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True,
                 ratio_per_sample=False,
                 *args, **kwargs):

        kwargs.pop('window_attn')
        kwargs.pop('window_size')
        assert kwargs.pop('layer_scale') is False, 'Pytorch Better Transformer does not support layer scale'
        assert kwargs.pop('pretrained') is None, 'use load_from instead of pretrained'
        super().__init__(num_heads=num_heads, *args, **kwargs)
        self.norm = None
        self.head = None



        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        if not self.inherit_mask:
            self.interactions = nn.Sequential(*[
                InteractionBlockWithSelection(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                 init_values=init_values, drop_path=self.drop_path_rate,
                                 norm_layer=self.norm_layer, with_cffn=with_cffn,
                                 cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                 extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                                 ratio_per_sample=ratio_per_sample)
                for i in range(len(interaction_indexes))
            ])
        else:
            self.interactions = nn.Sequential(*[
                InteractionBlockWithInheritSelection(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                                     init_values=init_values, drop_path=self.drop_path_rate,
                                                     norm_layer=self.norm_layer, with_cffn=with_cffn,
                                                     cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                                     extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                                                     ratio_per_sample=ratio_per_sample)
                for i in range(len(interaction_indexes))
            ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x, need_loss=False):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        if self.inherit_mask:
            prev_decision = torch.ones(bs, n, 1, device=x.device)
        ratio_loss = 0.
        num_loss = 0
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if not self.inherit_mask:
                x, c, layer_ratio_loss, has_loss = layer(x, c, indexes,
                             deform_inputs1, deform_inputs2, H, W,
                             self.blocks, self.selective_modules, self.keep_ratio)
            else:
                x, c, layer_ratio_loss, has_loss, prev_decision = layer(x, c, indexes,
                             deform_inputs1, deform_inputs2, H, W,
                             self.blocks, self.selective_modules, self.keep_ratio, self.sl_loc, prev_decision)
            ratio_loss = ratio_loss + layer_ratio_loss
            num_loss = num_loss + has_loss

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        if need_loss:
            return [f1, f2, f3, f4], {'backbone.ratio_loss': ratio_loss / num_loss}
        else:
            return [f1, f2, f3, f4]




# segmentor ======================================================================================

from .base_segmenter import BaseSegmenter, BaseSegmentor_Config
from ..decode_heads.segformer_head import SegformerHead

class SViT_Seg(BaseSegmenter):
    def __init__(self, 
                 backbone_cfg = None,
                 decode_head_cfg = None,
                 
                 
                 threshold: float = None, 
                 loss_decode=dict(loss_type='CrossEntropyLoss', 
                                  reduction='mean'), 
                 ignore_index: int = 255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold, loss_decode, ignore_index, align_corners)

        self.backbone = SelectiveViTAdapter(**backbone_cfg)
        self.decode_head = SegformerHead(**decode_head_cfg)
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')


    def forward(self, inputs: dict, need_loss = False):
        x = inputs['image']
        
        if need_loss:
            out_backbone, loss_dict = self.backbone(x, need_loss=need_loss)
            # loss_dict: {'backbone.ratio_loss': ratio_loss / num_loss}
        else:
            out_backbone = self.backbone(x, need_loss=need_loss)
            loss_dict = None
        
        logits_mask = self.decode_head(out_backbone)
        
        results = dict(
            logits_mask = logits_mask,
            ratio_loss = loss_dict
        )
        return results
    
    def loss(self, inputs: Dict[str, Tensor], labels: Dict[str, Tensor],
             return_logits: bool = False
             ) -> dict:
        """Forward function for training.

        Args:
            

        Returns:
            
        Shape:
            inputs: dict(
                image: (N, C, H, W)
            )
            
            labels: dict(
                label_mask: (N, out_channel, H, W)
            ) 
            
        """
        results = self.forward(inputs, need_loss=True)
        
        ratio_loss = results['ratio_loss']
        
        seg_logits = results['logits_mask']
        
        seg_label = labels['label_mask']
        
        logits_prob = torch.sigmoid(seg_logits) # for metric computing
        
        
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:], #(N, 1, H, W)
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)
        # (N, H, W)
        
        # Calculate loss
        losses = dict()
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
            
        else:
            losses_decode = self.loss_decode
        # losses_decode: loss layer(s) in Modulelist
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name.startswith('mask_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        )
            
            else:
                raise ValueError(f'loss name: {loss_decode.loss_name} is not supported')
                    
        losses['ratio_loss'] = ratio_loss['backbone.ratio_loss']
               
        # losses: {
        #         
        #         'loss_name1': loss_value1
        #         ...
        #     }
        
        preds = dict(pred_mask=logits_prob)
        
        if return_logits:
            return losses, preds
        else:
            return losses
    
    
    
    
class SViT_Seg_Config(BaseSegmentor_Config):
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'Full', 
                 
                 backbone_cfg: dict = None, 
                 decode_head_cfg: dict = None,
                 
                 threshold=None, 
                 loss_decode=None,
                 ignore_index=255, 
                 align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)
        
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
    def set_model_class(self):
        self.model_class = SViT_Seg



# Adapter modules ========================================================================================
# ========================================================================================================


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                                   (h // 16, w // 16),
                                                   (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
    
    
class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query
    

class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
            
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query

    
class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()
        
        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None
    
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c

    
class InteractionBlockWithSelection(InteractionBlock):
    def __init__(self, ratio_per_sample=False, **kwargs):
        super(InteractionBlockWithSelection, self).__init__(**kwargs)
        self.ratio_per_sample = ratio_per_sample


    def _ratio_loss(self, selector: torch.Tensor, ratio=1.):
        if not self.ratio_per_sample:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio)**2
        else:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio):
        n_skip = 3
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        layer_ratio_loss = 0.
        has_loss = 0
        for i in range(indexes[0], indexes[-1] + 1):
            if i < n_skip:
                x = blks[i](x)
            else:
                if self.training:
                    selector, diff_selector = selective_modules[i - n_skip](x)
                    x = diff_selector * blks[i](x, src_key_padding_mask=~selector) + \
                        (1 - diff_selector) * x
                    layer_ratio_loss += self._ratio_loss(diff_selector, keep_ratio[i - n_skip])
                    has_loss += 1
                else:
                    if x.shape[0] == 1:
                        selector, _ = selective_modules[i - n_skip](x)
                        real_indices = torch.argsort(selector.int(), dim=1, descending=True)\
                                        [:, :selector.sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        selected_x = torch.gather(x, 1, real_indices)
                        selected_x = blks[i](selected_x)
                        x.scatter_(1, real_indices, selected_x)
                    else:
                        selector, diff_selector = selective_modules[i - n_skip](x)
                        l_aligned_x, l_aligned_mask = left_align_tokens2(x, selector)
                        nt_x = torch._nested_tensor_from_mask(l_aligned_x, l_aligned_mask, mask_check=False)
                        nt_x = blks[i](nt_x, src_key_padding_mask=None)
                        x.masked_scatter_(selector.unsqueeze(-1), torch.cat(nt_x.unbind(), 0))

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, layer_ratio_loss, has_loss

    def forward_demo(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio):
        n_skip = 3
        # assert (blks[0].TransformerEncoderLayer.self_attn.num_heads % 2) == 0
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        sele_dict = {}
        for i in range(indexes[0], indexes[-1] + 1):
            if i < n_skip:
                x = blks[i](x)
            else:
                if self.training:
                    selector, diff_selector = selective_modules[i - n_skip](x)
                    x = diff_selector * blks[i](x, src_key_padding_mask=~selector) + \
                        (1 - diff_selector) * x
                else:
                    if x.shape[0] == 1:
                        selector, _ = selective_modules[i - n_skip](x)
                        real_indices = torch.argsort(selector.int(), dim=1, descending=True)[:,
                                       :selector.sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        selected_x = torch.gather(x, 1, real_indices)
                        selected_x = blks[i](selected_x)
                        x.scatter_(1, real_indices, selected_x)
                    else:
                        selector, diff_selector = selective_modules[i - n_skip](x)
                        l_aligned_x, l_aligned_mask = left_align_tokens2(x, selector)
                        nt_x = torch._nested_tensor_from_mask(l_aligned_x, l_aligned_mask, mask_check=False)
                        nt_x = blks[i](nt_x, src_key_padding_mask=None)
                        x.masked_scatter_(selector.unsqueeze(-1), torch.cat(nt_x.unbind(), 0))


                sele_dict[i] = selector

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, sele_dict




class InteractionBlockWithInheritSelection(InteractionBlock):
    def __init__(self, ratio_per_sample=False, **kwargs):
        super(InteractionBlockWithInheritSelection, self).__init__(**kwargs)
        self.ratio_per_sample = ratio_per_sample

    def _ratio_loss(self, selector: torch.Tensor, ratio=1.):
        if not self.ratio_per_sample:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio)**2
        else:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio, sl_loc,
                prev_decision):
        n_skip = 3
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        layer_ratio_loss = 0.
        has_loss = 0
        selector = None
        for i in range(indexes[0], indexes[-1] + 1):
            if i < n_skip:
                x = blks[i](x)
            else:
                if self.training:
                    if i+1 in sl_loc:
                        idx_convert = {3: 0, 6: 1, 9: 2}  # this is temporarily hard-coded
                        selector, diff_selector = selective_modules[idx_convert[i]](x)

                        # ------- added new code for correct inherit masks -------
                        diff_selector = diff_selector * prev_decision
                        prev_decision = diff_selector
                        selector = diff_selector.squeeze(-1).bool()
                        # --------------------------------------------------------

                        x = diff_selector * blks[i](x, src_key_padding_mask=~selector) + \
                            (1 - diff_selector) * x
                        layer_ratio_loss += self._ratio_loss(diff_selector, keep_ratio[idx_convert[i]])
                        has_loss += 1
                    else:
                        assert selector is not None
                        x = selector.float().unsqueeze(-1) * blks[i](x, src_key_padding_mask=~selector) + \
                            (1 - selector.float().unsqueeze(-1)) * x
                else:
                    if i+1 in sl_loc:
                        idx_convert = {3: 0, 6: 1, 9: 2}  # this is temporarily hard-coded
                        selector, _ = selective_modules[idx_convert[i]](x)

                        # ------- added new code for correct inherit masks -------
                        selector = selector * prev_decision.squeeze(-1)
                        prev_decision = selector.unsqueeze(-1)
                        # --------------------------------------------------------

                    if x.shape[0] == 1:
                        real_indices = torch.argsort(selector.int(), dim=1, descending=True)\
                            [:, :selector.long().sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        selected_x = torch.gather(x, 1, real_indices)
                        selected_x = blks[i](selected_x)
                        x.scatter_(1, real_indices, selected_x)
                    else:
                        l_aligned_x, l_aligned_mask = left_align_tokens2(x, selector)
                        nt_x = torch._nested_tensor_from_mask(l_aligned_x, l_aligned_mask, mask_check=False)
                        nt_x = blks[i](nt_x, src_key_padding_mask=None)
                        x.masked_scatter_(selector.unsqueeze(-1), torch.cat(nt_x.unbind(), 0))


        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, layer_ratio_loss, has_loss, prev_decision

    def forward_demo(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio, sl_loc,
                     prev_decision):
        n_skip = 3
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        sele_dict={}
        selector = None
        for i in range(indexes[0], indexes[-1] + 1):
            if i < n_skip:
                x = blks[i](x)
            else:
                if i+1 in sl_loc:
                    idx_convert = {3: 0, 6: 1, 9: 2}  # this is temporarily hard-coded
                    selector, diff_selector = selective_modules[idx_convert[i]](x)

                    # ------- added new code for correct inherit masks -------
                    diff_selector = diff_selector * prev_decision
                    prev_decision = diff_selector
                    selector = diff_selector.squeeze(-1).bool()
                    # --------------------------------------------------------

                    x = diff_selector * blks[i](x, src_key_padding_mask=~selector) + \
                        (1 - diff_selector) * x
                else:
                    assert selector is not None
                    x = selector.float().unsqueeze(-1) * blks[i](x, src_key_padding_mask=~selector) + \
                        (1 - selector.float().unsqueeze(-1)) * x
                sele_dict[i] = selector


        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, sele_dict, prev_decision
