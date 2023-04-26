# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class MaskFeature(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""
        inputs = inputs[0]
        mask = torch.where(torch.rand_like(inputs) > 0.5, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
        mask_feature = torch.mul(inputs, mask)
        
        return inputs, mask_feature
