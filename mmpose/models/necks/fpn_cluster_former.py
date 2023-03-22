# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import (build_norm_layer, constant_init, normal_init,
                      trunc_normal_init)
from mmcv.runner import _load_checkpoint, load_state_dict

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import (PatchEmbed, TCFormerDynamicBlock, TCFormerRegularBlock,
                     TokenConv, cluster_dpc_knn, merge_tokens,
                     tcformer_convert, token2map, map2token)
from .gap_neck import GlobalAveragePooling

class CTM(nn.Module):
    """Clustering-based Token Merging module in TCFormer.

    Args:
        sample_ratio (float): The sample ratio of tokens.
        embed_dim (int): Input token feature dimension.
        dim_out (int): Output token feature dimension.
        k (int): number of the nearest neighbor used i DPC-knn algorithm.

    return:
        donw_dict:tuple(dict_new ,dict_old).
        dict_ has same attribute at token_dict in TCFormer
        dict_new['token_num'] = 0.25(ratio) * dict_old['token_num']
    """

    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(
            in_channels=embed_dim,
            out_channels=dim_out,
            kernel_size=3,
            stride=2,
            padding=1)
        # self.conv = nn.Conv1d(
        #     in_channels=embed_dim,
        #     out_channels=dim_out,
        #     kernel_size=1,
        #     bias=False)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        x = self.conv(token_dict)
        # x = token_dict['x']
        # x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.exp()

        token_dict['x'] = x
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        idx_cluster, cluster_num = cluster_dpc_knn(token_dict, cluster_num,
                                                   self.k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num,
                                 token_weight)

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        return down_dict, token_dict


@BACKBONES.register_module()
class FpCFormer(nn.Module):
    """Fpn + Token Clustering Transformer (FpFormer)

    Implementation of `Not All Tokens Are Equal: Human-centric Visual
    Analysis via Token Clustering Transformer
    <https://arxiv.org/abs/2204.08680>`

        Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (list[int]): Embedding dimension. Default:
            [64, 128, 256, 512].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer block.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', eps=1e-6).
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer block. Default: [8, 4, 2, 1].
        num_stages (int): The num of stages. Default: 4.
        pretrained (str, optional): model pretrained path. Default: None.
        k (int): number of the nearest neighbor used for local density.
        sample_ratios (list[float]): The sample ratios of CTM modules.
            Default: [0.25, 0.25, 0.25]
        return_map (bool): If True, transfer dynamic tokens to feature map at
            last. Default: False
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
    """

    def __init__(self,
                 in_channels=[256,512,1024,2048],
                 embed_dims=[256,512,1024,2048],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 num_layers=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 pretrained=None,
                 k=5,
                 sample_ratios=[0.25, 0.25, 0.25],
                 # sample_ratios=[0.5, 0.5, 0.5],
                 return_map=False,
                 convert_weights=True):
        super().__init__()

        self.num_layers = num_layers
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.convert_weights = convert_weights


        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]
        cur = 0

        # backbone 1~4 stage to sample channel though 1x1Conv2d
        # for i,in_chan in enumerate(self.in_channels):
        #     conv = nn.Conv2d(in_chan, in_chan//4, 1)
        #     setattr(self, f'conv1{i + 1}', conv)

        # In stage 1, use the standard transformer blocks
        for i in range(1):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims[i],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_cfg=dict(type='LN', eps=1e-6))

            block = nn.ModuleList([
                TCFormerRegularBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]

            cur += num_layers[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        # In stage 2~4, use TCFormerDynamicBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i - 1], embed_dims[i - 1], embed_dims[i],
                      k)

            ups = nn.Upsample(scale_factor=2**i, mode='bilinear')

            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims[i],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_cfg=dict(type='LN', eps=1e-6))

            block = nn.ModuleList([
                TCFormerDynamicBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            cur += num_layers[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'ctm{i}', ctm)
            setattr(self, f'ups{i}', ups)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        self.gap = GlobalAveragePooling()
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()

            checkpoint = _load_checkpoint(
                pretrained, logger=logger, map_location='cpu')
            logger.warning(f'Load pre-trained model for '
                           f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if self.convert_weights:
                # We need to convert pre-trained weights to match this
                # implementation.
                state_dict = tcformer_convert(state_dict)
            load_state_dict(self, state_dict, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # outs = []
        temp = []
        shp = []
        # init token embedding of each stage
        for j in range(4):
            patch_embed = getattr(self, f'patch_embed{j + 1}')
            tem, (H, W) = patch_embed(x[j])# backbone中同一层的特征转为token输入
            temp.append(tem)
            shp.append((H, W))

        x1 = temp[0]
        H, W = shp[0]
        # print(x1.shape)

        i = 0
        block = getattr(self, f'block{i + 1}')#fpcformer_block[]
        norm = getattr(self, f'norm{i + 1}')
        # x, (H, W) = patch_embed(x)
        for blk in block:
            x1 = blk(x1, H, W)
        x1 = norm(x1)

        # init token dict
        B, N, _ = x1.shape
        device = x1.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x1.new_ones(B, N, 1)
        token_dict = {
            'x': x1,
            'token_num': N,
            'map_size': [H, W],
            'init_grid_size': [H, W], # for upsample to this size
            'idx_token': idx_token, # for upsample about the project
            'agg_weight': agg_weight # importance weight
        }
        # outs.append(token_dict.copy())

        # stage 2~4
        for i in range(1, self.num_stages):
            # print('stage = ', i)
            ctm = getattr(self, f'ctm{i}')
            ups = getattr(self, f'ups{i}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            # print('pre x.shape = ',token_dict['x'].shape)
            token_dict = ctm(token_dict)  # down sample
            ## through the ctm, token_dict is a tuple(dict_new, dict_old)
            ## so the next step needs to token_dict[0] nor token_dict
            ## ctm 内部通过1x1卷积将上一个stage通道翻倍，刚好匹配下一个stage的token的dim数
            B, N, C = temp[i].shape
            H, W = shp[i]
            temp[i] = temp[i].permute(0,2,1).reshape(B,C,H,W)
            temp[i] = ups(temp[i])
            token_dict[0]['x'] += map2token(temp[i], token_dict[0])  # feature fusion 没写错
            # print('aft x.shape = ', token_dict[0]['x'].shape)
            # token_dict['x'] = norm(token_dict['x']) # test norm
            for j, blk in enumerate(block):
                token_dict = blk(token_dict)

            token_dict['x'] = norm(token_dict['x'])
            # outs.append(token_dict)
        if self.return_map:
            B, N, C = token_dict['x'].shape
            H, W = shp[3]
            outs = token_dict['x'].permute(0,2,1).reshape(B,C,H,W)
            outs = self.gap(outs)
            # outs = [token2map(token_dict) for token_dict in outs]

        return outs
