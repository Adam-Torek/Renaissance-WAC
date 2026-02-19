from .base_dataset import BaseDataset
import io
from PIL import Image
import torchvision.transforms.functional as F
import torch
import random
import numpy as np
import pyarrow as pa
import pandas as pd

class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
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

        self.subimages_format = "all"
        if "loss_names" in kwargs:
            loss_names = kwargs.pop("loss_names")
            if loss_names["ref_bbox"] > 0:
                self.subimages_format = "bboxes"
            if loss_names["itm"] > 0:
                self.subimages_format = "none"

    def get_bounding_boxes(self, index, only_one=False, return_normalized=False):
        text_index, _ = self.index_mapper[index]
        bboxes = self.table["bboxes"][text_index].as_py()
        if only_one:
            label = self.table["labels"][text_index].as_py()
            bboxes = [bboxes[label]]
        
        if return_normalized:
            width, height = self.get_raw_image(index).size
            for i, bbox in enumerate(bboxes):
                bbox_center = [bbox[2]/2, bbox[3]/2, bbox[2], bbox[3]]
                bbox_normalized = [bbox_center[0]/width, bbox_center[1]/height, bbox_center[2]/width, bbox_center[3]/height]
                bboxes[i] = bbox_normalized
        
        return bboxes        

    def get_subimages(self, index, only_one=False):
        image_data = self.get_raw_image(index)
        bboxes = self.get_bounding_boxes(index, only_one=only_one)

        subimages_list = []
        for bbox in bboxes:
            bbox_bounds = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

            cropped_image = image_data.crop(bbox_bounds)

            resized_subimage = cropped_image.resize((self.image_size, self.image_size), resample=Image.Resampling.LANCZOS)
            
            resized_subimage = [tr(resized_subimage) for tr in self.transforms][0]

            subimages_list.append(resized_subimage)
        
        return subimages_list

    def __getitem__(self, index):

        return_dict = {}
        
        text_data = self.get_text(index)
        return_dict.update(text_data)

        if self.subimages_format == "bboxes":
            bboxes_normalized = self.get_bounding_boxes(index, only_one=False, return_normalized=True)
            return_dict["bboxes"] = bboxes_normalized

        elif self.subimages_format == "all" or self.include_wac_data:
            subimages = self.get_subimages(index, only_one=False)
            return_dict["subimages"] = torch.stack(subimages)
            
        elif self.subimages_format == "none":
            return_dict["image"] = self.get_subimages(index, only_one=True)[0]
            for i in range(0, self.draw_false_image):
                random_index = random.randint(0, len(self.index_mapper))
                random_subimage = self.get_subimages(random_index, only_one=True)[0]
                return_dict[f"false_image_{i}"] = random_subimage

            for i in range(0, self.draw_false_text):
                random_index = random.randint(0, len(self.index_mapper))
                random_text = self.get_text(random_index)
                return_dict[f"false_text_{i}"] = random_text
        
        # Get subimage and position data for WAC models if needed
        if self.include_wac_data:

            # Gather position data to supplement WAC models with location information
            bounding_boxes = self.get_bounding_boxes(index)
            image_width, image_height = self.get_raw_image(index).size
            bbox_position_list = []
            for bbox in bounding_boxes:
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

                bbox_position_data = np.array([x_1_rel, y_1_rel, x_2_rel, y_2_rel, rel_area, side_ratio, distance_from_center])
                bbox_position_list.append(bbox_position_data)

            bbox_position_list = np.stack(bbox_position_list)
            text = text_data["text"][0]
            tokenized_text = self.tokenizer.tokenize(text)
            return_dict["tokenized_words"] = tokenized_text
            return_dict["position_data"] = bbox_position_list
       
        text_index, _ = self.index_mapper[index]
        return_dict["label"] = torch.tensor(self.table["labels"][text_index].as_py())
        return_dict["ann_id"] = self.table["ann_ids"][text_index].as_py()
        return_dict["num_objects"] = torch.tensor(self.table["num_objects"][text_index].as_py())

        return return_dict

    def torch_collate(self, dict_batch, batch_items, key):
        dict_batch[key] = torch.stack([batch_item[key] for batch_item in batch_items])
        return dict_batch

    def list_collate_if_exists(self, dict_batch, batch_items, key):
        if key in batch_items[0]:
            dict_batch[key] = [batch_item[key] for batch_item in batch_items]
        return dict_batch

    def collate(self, batch, mlm_collator=None):
        dict_batch = {}
        dict_batch["text_ids"] = []
        dict_batch["text"] = []

        
        dict_batch = self.torch_collate(dict_batch, batch, "label")
        dict_batch = self.torch_collate(dict_batch, batch, "num_objects")

        dict_batch = self.list_collate_if_exists(dict_batch, batch, "subimages")
        dict_batch = self.list_collate_if_exists(dict_batch, batch, "position_data")
        dict_batch = self.list_collate_if_exists(dict_batch, batch, "tokenized_words")
        dict_batch = self.list_collate_if_exists(dict_batch, batch, "position_data")
        dict_batch = self.list_collate_if_exists(dict_batch, batch, "ann_id")

        for batch_item in batch:
            batch_text_pair = batch_item["text"]
            batch_text_data = batch_text_pair[1]
            batch_text = batch_text_pair[0]
            
            dict_batch["text"].append(batch_text)
            dict_batch["text_ids"].append(torch.tensor(batch_text_data["input_ids"]))
            if "attention_mask" in batch_text_data:
                if "text_masks" not in dict_batch:
                    dict_batch["text_masks"] = []
                dict_batch["text_masks"].append(torch.tensor(batch_text_data["attention_mask"]))

        dict_batch["text_ids"] = torch.stack(dict_batch["text_ids"])

        if "text_masks" in dict_batch:
            dict_batch["text_masks"] = torch.stack(dict_batch["text_masks"])

        if "image" in batch[0]:
            images_tensor = torch.stack([batch_item["image"] for batch_item in batch])
            dict_batch["image"] = [images_tensor]

        if self.draw_false_image > 0:
            for i in range(0, self.draw_false_image):
                false_images_tensor = torch.stack([batch_item[f"false_image_{i}"] for batch_item in batch])
                dict_batch[f"false_image_{i}"] = [false_images_tensor]

        return dict_batch
                    