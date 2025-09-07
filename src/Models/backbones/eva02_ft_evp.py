
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mmengine.utils import to_2tuple


from ..utils.drop import DropPath
from ..utils.rope import VisionRotaryEmbeddingFast
from ..utils import (build_activation_layer, build_dropout, build_norm_layer,
                     nlc_to_nchw, nchw_to_nlc)

from .eva02 import (Mlp, SwiGLU, PatchEmbed, 
                  RelativePositionBias, DecoupledRelativePositionBias, 
                  Attention, Block)

    
    



class EVA02EVP(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768,
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                **kwargs
                ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim 
        
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu
        
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.embed_dims = [embed_dim, ]
        self.depths = [depth, ]
        # vpt config
        self.scale_factor = kwargs['scale_factor']
        self.prompt_type = kwargs['prompt_type']
        self.tuning_stage = str(kwargs['tuning_stage'])
        self.input_type = kwargs['input_type']
        self.freq_nums = kwargs['freq_nums']
        self.handcrafted_tune = kwargs['handcrafted_tune']
        self.embedding_tune = kwargs['embedding_tune']
        self.adaptor = kwargs['adaptor']
        self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dims, self.tuning_stage, self.depths,
                                                self.input_type, self.freq_nums,
                                                self.handcrafted_tune, self.embedding_tune, self.adaptor,
                                                img_size)

    

    

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    

    def forward_features(self, x):
        # x = inputs['image']
        
        if self.handcrafted_tune:
            handcrafted1 = self.prompt_generator.init_handcrafted(x)
        else:
            handcrafted1 = None
        
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        # outs = []
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # handcrafted1 = torch.cat((cls_tokens, handcrafted1), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        # img_feat = x[:, 1:, :]

        

        # stage 1
        # x, (H, W) = self.patch_embed1(x)
        if '1' in self.tuning_stage:
            prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.blocks):
            if '1' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x, rel_pos_bias)
        x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

       

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x




# =============================================================================================
# ================================== EVP ======================================================

class GaussianFilter(nn.Module):
    def __init__(self):
        super(GaussianFilter, self).__init__()
       
    

    def conv_gauss(self, img):
        def gauss_kernel(channels=3, device=...):
            kernel = torch.tensor([[1., 4., 6., 4., 1],
                                [4., 16., 24., 16., 4.],
                                [6., 24., 36., 24., 6.],
                                [4., 16., 24., 16., 4.],
                                [1., 4., 6., 4., 1.]])
            kernel /= 256.
            kernel = kernel.repeat(channels, 1, 1, 1)
            kernel = kernel.to(device)
            return kernel
        
        kernel = gauss_kernel(device=img.device)
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out


class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        self.srm_layer = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,)
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 2 / 4, -4 / 4, 2 / 4, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1 / 2, -2 / 2, 1 / 2, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        self.srm_layer.weight.data = torch.Tensor(
            [[filter1, filter1, filter1],
             [filter2, filter2, filter2],
             [filter3, filter3, filter3]]
        )

        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def conv_srm(self, img):
        out = self.srm_layer(img)
        return out


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * \
#             (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim,
#                               kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

#         # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x


class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dims, tuning_stage, depths, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depths = depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        if self.input_type == 'gaussian':
            self.gaussian_filter = GaussianFilter()
        if self.input_type == 'srm':
            self.srm_filter = SRMFilter()
        if self.input_type == 'all':
            self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size), requires_grad=False)

        if self.handcrafted_tune:
            if '1' in self.tuning_stage:
                # self.handcrafted_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                #                                         embed_dim=self.embed_dims[0] // self.scale_factor)
                self.handcrafted_generator1 = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, 
                                                         embed_dim=self.embed_dims[0] // self.scale_factor)
                self.register1 = nn.Parameter(torch.zeros(1, 1, self.embed_dims[0] // self.scale_factor))
            # if '2' in self.tuning_stage:
            #     self.handcrafted_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
            #                                            in_chans=self.embed_dims[0] // self.scale_factor,
            #                                            embed_dim=self.embed_dims[1] // self.scale_factor)
            # if '3' in self.tuning_stage:
            #     self.handcrafted_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
            #                                            in_chans=self.embed_dims[1] // self.scale_factor,
            #                                            embed_dim=self.embed_dims[2] // self.scale_factor)
            # if '4' in self.tuning_stage:
            #     self.handcrafted_generator4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
            #                                            in_chans=self.embed_dims[2] // self.scale_factor,
            #                                            embed_dim=self.embed_dims[3] // self.scale_factor)

        if self.embedding_tune:
            if '1' in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor)

        if self.adaptor == 'adaptor':
            if '1' in self.tuning_stage:
                for i in range(self.depths[0]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

            if '2' in self.tuning_stage:
                for i in range(self.depths[1]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])

            if '3' in self.tuning_stage:
                for i in range(self.depths[2]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])

            if '4' in self.tuning_stage:
                for i in range(self.depths[3]):
                    lightweight_mlp = nn.Sequential(
                            nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                            nn.GELU(),
                        )
                    setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])

        elif self.adaptor == 'fully_shared':
            self.fully_shared_mlp1 = nn.Sequential(
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                    )
            self.fully_shared_mlp2 = nn.Sequential(
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                    )
            self.fully_shared_mlp3 = nn.Sequential(
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                    )
            self.fully_shared_mlp4 = nn.Sequential(
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                        nn.GELU(),
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                    )

        elif self.adaptor == 'fully_unshared':
            for i in range(self.depths[0]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
                )
                setattr(self, 'fully_unshared_mlp1_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[1]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
                )
                setattr(self, 'fully_unshared_mlp2_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[2]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
                )
                setattr(self, 'fully_unshared_mlp3_{}'.format(str(i)), fully_unshared_mlp1)
            for i in range(self.depths[3]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                    nn.GELU(),
                    nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
                )
                setattr(self, 'fully_unshared_mlp4_{}'.format(str(i)), fully_unshared_mlp1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_handcrafted(self, x):
        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums, self.prompt_type)

        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        elif self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x, device=x.device)

        elif self.input_type == 'srm':
            x = self.srm_filter.srm_layer(x)

        # return x
        B = x.shape[0]
        # get prompting

        if '1' in self.tuning_stage:
            handcrafted1 = self.handcrafted_generator1(x)
            register_tokens = self.register1.expand(B, -1, -1)
            handcrafted1 = torch.cat((register_tokens, handcrafted1), dim=1)
        else:
            handcrafted1 = None

        # if '2' in self.tuning_stage:
        #     handcrafted2, H2, W2 = self.handcrafted_generator2(handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
        # else:
        #     handcrafted2 = None

        # if '3' in self.tuning_stage:
        #     handcrafted3, H3, W3 = self.handcrafted_generator3(handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
        # else:
        #     handcrafted3 = None

        # if '4' in self.tuning_stage:
        #     handcrafted4, H4, W4 = self.handcrafted_generator4(handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
        # else:
        #     handcrafted4 = None

        return handcrafted1

    def init_prompt(self, embedding_feature, handcrafted_feature, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(embedding_feature)
        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature

        return handcrafted_feature, embedding_feature

    def get_embedding_feature(self, x, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(x)

            return embedding_feature
        else:
            return None

    def get_handcrafted_feature(self, x, block_num):
        if self.handcrafted_tune:
            handcrafted_generator = getattr(self, 'handcrafted_generator{}'.format(str(block_num)))
            handcrafted_feature = handcrafted_generator(x)

            return handcrafted_feature
        else:
            return None

    def get_prompt(self, x, prompt, block_num, depth_num):
        feat = 0
        if self.handcrafted_tune:
            feat += prompt[0]
        if self.embedding_tune:
            feat += prompt[1]

        if self.adaptor == 'adaptor':
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
            shared_mlp = getattr(self, 'shared_mlp' + str(block_num))
            

            feat = lightweight_mlp(feat)
            feat = shared_mlp(feat)

        elif self.adaptor == 'fully_shared':
            fully_shared_mlp = getattr(self, 'fully_shared_mlp' + str(block_num))
            feat = fully_shared_mlp(feat)

        elif self.adaptor == 'fully_unshared':
            fully_unshared_mlp = getattr(self, 'fully_unshared_mlp' + str(block_num) + '_' + str(depth_num))
            feat = fully_unshared_mlp(feat)

        x = x + feat

        return x

    def fft(self, x, rate, prompt_type):
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        if prompt_type == 'highpass':
            fft = fft * (1 - mask)
        elif prompt_type == 'lowpass':
            fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv












