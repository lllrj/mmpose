# Copyright (c) OpenMMLab. All rights reserved.
from .ae_head import AssociativeEmbeddingHead
from .cid_head import CIDHead
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .mspn_head import MSPNHead
from .vipnas_head import ViPNASHead
from .maskheatmap_head import MaskHeatmapHead
__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead', 'MaskHeatmapHead'
]
