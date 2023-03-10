from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset

@DATASETS.register_module()
class AirDataset(BaseCocoStyleDataset):
    """CocoDataset dataset for top-down pose estimation.

    "Microsoft COCO: Common Objects in Context", ECCV'2014.
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Air keypoint indexes::

        0:"head",#机头
        1:"left_inner_wing",#左侧机翼内侧拐点（与机身交点）
        2:"left_out_wing",#左侧机翼外侧顶点
        3:"left_horizontal_inner_wing",#左后侧机翼内侧拐点（与机身交点）
        4:"left_horizontal_out_wing",#左后侧机翼外侧顶点
        5:"right_inner_wing",#右侧机翼内侧拐点（与机身交点）
        6:"right_out_wing",,#右侧机翼外侧顶点
        7:"right_horizontal_inner_wing",#右后侧机翼内侧拐点（与机身交点）
        8:"right_horizontal_out_wing",#右后侧机翼外侧顶点
        9:"lower_vertical_wing",,#竖直尾翼下侧顶点
        10:"upper_vertical_wing",#竖直尾翼上侧顶点
        11:"fuselage"#机身中点

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """
    METAINFO: dict = dict(from_file='configs/_base_/datasets/Air.py')