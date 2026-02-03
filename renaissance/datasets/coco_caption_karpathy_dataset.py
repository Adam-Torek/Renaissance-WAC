from .base_dataset import BaseDataset
import io
from PIL import Image

class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", include_wac_data=True, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["refcoco_unc_train"]
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            names = ["refcoco_unc_val"]
        elif split == "test":
            names = ["refcoco_unc_test"]

        super().__init__(*args, **kwargs, names=names, include_wac_data=include_wac_data, text_column_name="sentences")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        # if "test" in self.split:
        #     _index, _question_index = self.index_mapper[index]
        #     iid = self.table["image_id"][_index].as_py()
        #     iid = int(iid.split(".")[0].split("_")[-1])
        #     suite.update({"iid": iid})

        text_index, _ = self.index_mapper[index]
        suite["label"] = self.table["labels"][text_index].as_py()
        suite["ann_id"] = self.table["ann_ids"][text_index].as_py()

        return suite
