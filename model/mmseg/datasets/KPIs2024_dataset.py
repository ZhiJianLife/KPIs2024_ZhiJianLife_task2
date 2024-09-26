# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class KPIs2024Dataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'Kindey')

    PALETTE = [[0,0,0], [255,255,255]]

    def __init__(self, **kwargs):
        super(KPIs2024Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            classes = ('background', 'Kindey'),
            palette =  [[0,0,0], [255,255,255]],
            **kwargs)
