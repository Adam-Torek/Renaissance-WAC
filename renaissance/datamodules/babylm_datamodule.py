from renaissance.datamodules.datamodule_base import BaseDataModule
from renaissance.datasets.babylm_dataset import BabyLMDataset


class BabyLMDataModule(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def dataset_cls(self):
        return BabyLMDataset
    
    @property
    def dataset_cls_no_false(self):
        return BabyLMDataset
    
    @property
    def dataset_name(self):
        return "babylm"