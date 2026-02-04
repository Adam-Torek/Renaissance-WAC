from .base_dataset import BaseDataset
import io
from PIL import Image
import torchvision.transforms.functional as F
import torch
import random

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

        super().__init__(*args, **kwargs, names=names, text_column_name="sentences")

        self.include_wac_data = False 

        if self.draw_false_image == 0 or self.draw_false_text == 0 or not self.image_only:
            if "include_wac_data" in kwargs:
                include_wac_data = kwargs.pop("include_wac_data")
                self.include_wac_data = include_wac_data
            else:
                self.include_wac_data = False

        self.is_ref_bbox = False
        if "loss_names" in kwargs:
            loss_names = kwargs.pop("loss_names")
            if loss_names["ref_bbox"] > 0:
                self.is_ref_bbox = True


    def get_subimage(self, index):
        text_index, _ = self.index_mapper[index]
        image_data = self.get_raw_image(index)
        width, height = image_data.size

        label = self.table["labels"][text_index].as_py()
        bbox = self.table["bboxes"][text_index][label].as_py()
        
        bbox_bounds = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        bbox_center = [bbox[2]/2, bbox[3]/2, bbox[2], bbox[3]]
        bbox_normalized = [bbox_center[0]/width, bbox_center[1]/height, bbox_center[2]/width, bbox_center[3]/height]

        cropped_image = image_data.crop(bbox_bounds)

        resized_subimage = cropped_image.resize((self.image_size, self.image_size), resample=Image.Resampling.LANCZOS)
        
        resized_subimage =  [tr(resized_subimage) for tr in self.transforms]
        return (resized_subimage, bbox, bbox_normalized)

    def __getitem__(self, index):

        return_dict = {}
        
        text_data = self.get_text(index)

       
        subimage, bbox, bbox_normalized = self.get_subimage(index)
        if not self.is_ref_bbox:
            return_dict["image"] = subimage
        else:
            image_tensor = self.get_raw_image(index)
            image_tensor = [tr(image_tensor) for tr in self.transforms]
            return_dict["image"] = image_tensor

        return_dict.update(text_data)
        
        for i in range(0, self.draw_false_image):
            random_index = random.randint(0, len(self.index_mapper) - 1)
            if not self.is_ref_bbox:
                false_subimage, _, _ = self.get_subimage(random_index)
            else:
                false_subimage = self.get_raw_image(random_index)
                false_subimage = [tr(false_subimage) for tr in self.transforms]
            
            return_dict[f"false_image_{i}"] = false_subimage
            
        for i in range(0, self.draw_false_text):
            return_dict.update(self.get_false_text(rep=f"{i}"))
        
        # Get subimage and position data for WAC models if needed
        if self.include_wac_data:

            # Gather position data to supplement WAC models with location information
            image_width, image_height = image.shape
            _, _, bbox_w, bbox_h = bbox
            bbox_x = bbox[0]
            bbox_y = bbox[1]
            bbox_x_2 = bbox[0] + bbox[2]
            bbox_y_2 = bbox[1] + bbox[3]

            x_1_rel, y_1_rel = bbox_x / image_width, bbox_y / image_height
            x_2_rel, y_2_rel = bbox_x_2 / image_width, bbox_y_2 / image_height

            center_x, center_y = image_width / 2, image_height / 2
            bbox_center_x, bbox_center_y = bbox_x_2 / image_width, bbox_y_2 / image_height

            rel_area = (bbox_w * bbox_h) / (image_width * image_height)
            side_ratio = image_width / image_height

            distance_from_center = np.sqrt((bbox_center_x-center_x)**2 + (bbox_center_y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)

            bbox_position_data = torch.tensor([x_1_rel, y_1_rel, x_2_rel, y_2_rel, rel_area, side_ratio, distance_from_center])

            tokenized_text = self.tokenizer.tokenize(text)
            return_dict["tokenized_words"] = tokenized_text
            return_dict["subimage"] = subimage
            return_dict["position_data"] = bbox_position_data

        if self.is_ref_bbox:
            return_dict["bbox_label"] = torch.tensor(bbox_normalized)
       
        text_index, _ = self.index_mapper[index]
        return_dict["label"] = torch.tensor(self.table["labels"][text_index].as_py())
        return_dict["ann_id"] = torch.tensor(self.table["ann_ids"][text_index].as_py())

        return return_dict
