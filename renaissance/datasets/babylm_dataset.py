

from .base_dataset import BaseDataset


class BabyLMDataset(BaseDataset):

    def __init__(self) -> None:
        pass

    def __getitem__(self) -> dict:
        return {}
    
    def __len__(self) -> int:
        return -1