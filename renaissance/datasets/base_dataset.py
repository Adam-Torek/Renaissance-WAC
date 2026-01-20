import random
import torch
import io
import pyarrow as pa
import os
import numpy as np

from PIL import Image 
from PIL.Image import Resampling
from ..transforms import keys_to_transforms
from datasets import load_dataset
from torch.utils.data._utils.collate import default_collate


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir = "",
        transform_keys = [],
        image_size = 224,
        names = [],
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        # text_only=False,
        hugging_face=False,
        hf_dataset_key = '',
        task = '',
        tokenizer=None,
        processor=None,
        include_wac_data=True,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        if not hugging_face:
            assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.clip_transform = False
        for transform_key in transform_keys:
            if 'clip' in transform_key:
                self.clip_transform = True
                break
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        
        self.image_only = image_only
        # self.text_only = text_only
        self.data_dir = data_dir
        
        self.image_size = image_size
        self.tokenizer = tokenizer
        # Suppresses a warning associated with the tokenizer
        # from https://github.com/huggingface/transformers/issues/22638
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.processor = processor
        
        self.hugging_face = hugging_face
        # self.task = 

        if draw_false_image > 0 or draw_false_text > 0 or self.image_only:
            self.include_wac_data = False
        else:
            self.include_wac_data = include_wac_data
        
        # Use hugging face dataset classes to download, load and manage data
        if self.hugging_face:
            if self.split=='val':
                self.split = 'validation'
            self.data_dict = load_dataset(hf_dataset_key, self.task, split=self.split).to_dict()
        # Use local files with pyarrow for data processing and loading
        else:
            if len(names) != 0:
                tables = [
                    pa.ipc.RecordBatchFileReader(
                        pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                    ).read_all()
                    for name in names
                    if os.path.isfile(f"{data_dir}/{name}.arrow")
                ]
    
                self.table_names = list()
                for i, name in enumerate(names):
                    self.table_names += [name] * len(tables[i])
    
                self.table = pa.concat_tables(tables)
                if text_column_name != "":
                    self.text_column_name = text_column_name
                    self.all_texts = self.table[text_column_name].to_pandas().tolist()
                    if type(self.all_texts[0][0]) == str:
                        self.all_texts = (
                            [list(set(texts)) for texts in self.all_texts]
                            if remove_duplicate
                            else self.all_texts
                        )
                    else: #snli
                        self.all_texts = (
                            [[t[1].strip() for t in texts] for texts in self.all_texts]
                        )
                else:
                    self.all_texts = list()
            else:
                self.all_texts = list()
    
            self.index_mapper = dict()
    
            if text_column_name != "" and not self.image_only:
                j = 0
                for i, texts in enumerate(self.all_texts):
                    for _j in range(len(texts)):
                        self.index_mapper[j] = (i, _j)
                        j += 1
            else:
                for i in range(len(self.table)):
                    self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        # if self.clip_transform:
        #     return Image.open(image_bytes).convert("RGBA")
        # else:
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)

        # Get the right bounding box using an index mapper to a text-image pair index
        if "bboxes" in self.table.column_names:
            text_index, _ = self.index_mapper[index]
            label = self.table["labels"][text_index].as_py()
            bbox = self.table["bboxes"][text_index][label].as_py()
            bbox_torch = torch.tensor(bbox)
        else: 
            bbox_torch = None

        # Get subimage and position data for WAC models if needed
        if self.include_wac_data and bbox_torch is not None:

            # Extract the subimage using the bounding box and use it to 
            bbox_int = [int(bound) for bound in bbox]
            bbox_bounds = [bbox_int[0], bbox_int[1], bbox_int[0]+bbox_int[2], bbox_int[1]+bbox_int[3]]
            subimage = image.crop(bbox_bounds)
            subimage = subimage.resize(image.size, resample=Resampling.LANCZOS)

            # Gather position data to supplement WAC models with location information
            image_width, image_height = image.size
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            bbox_x_2, bbox_y_2 = bbox_x+bbox_w, bbox_y + bbox_h

            x_1_rel, y_1_rel = bbox_x / image_width, bbox_y / image_height
            x_2_rel, y_2_rel = bbox_x_2 / image_width, bbox_y_2 / image_height

            center_x, center_y = image_width / 2, image_height / 2
            bbox_center_x, bbox_center_y = bbox_x_2 / image_width, bbox_y_2 / image_height

            rel_area = (bbox_w * bbox_h) / (image_width * image_height)
            side_ratio = image_width / image_height

            distance_from_center = np.sqrt((bbox_center_x-center_x)**2 + (bbox_center_y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)

            bbox_position_data = torch.tensor([x_1_rel, y_1_rel, x_2_rel, y_2_rel, rel_area, side_ratio, distance_from_center])
        else:
            subimage = None
            bbox_position_data = None

        image_tensor = [tr(image) for tr in self.transforms]
        if subimage is not None:
            subimage_tensor = [tr(image) for tr in self.transforms]
        else:
            subimage_tensor = None

        return_dict = {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

        if bbox_torch is not None:
            return_dict["bounding_box"] = bbox_torch

        if bbox_position_data is not None:
            return_dict["position_data"] = bbox_position_data

        if subimage_tensor is not None:
            return_dict["subimage"] = subimage_tensor

        return return_dict

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]

        return_dict = {}
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            # return_tensors='pt'
        )

        if self.include_wac_data:
            tokenized_text = self.tokenizer.tokenize(text)
            return_dict["tokenized_words"] = tokenized_text
        
        return_dict["text"] = (text, encoding)
        return_dict["img_index"] = index
        return_dict["cap_index"] = caption_index
        return_dict["raw_index"] = raw_index
        
        return return_dict

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        # collate function for locally stored datasets
        if not self.hugging_face:
        
            if isinstance(batch[0], dict) and "text" in batch[0] and isinstance(batch[0]["text"], tuple):
                batch_size = len(batch)
                keys = set([key for b in batch for key in b.keys()])
                dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

                img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
                img_sizes = list()
        
                for img_key in img_keys:
                    img = dict_batch[img_key]
                    img_sizes += [ii.shape for i in img if i is not None for ii in i]
        
                for size in img_sizes:
                    assert (
                        len(size) == 3
                    ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"
        
                if len(img_keys) != 0:
                    max_height = max([i[1] for i in img_sizes])
                    max_width = max([i[2] for i in img_sizes])
        
                for img_key in img_keys:
                    img = dict_batch[img_key]
                    view_size = len(img[0])
        
                    new_images = [
                        torch.zeros(batch_size, 3, max_height, max_width)
                        for _ in range(view_size)
                    ]
        
                    for bi in range(batch_size):
                        orig_batch = img[bi]
                        for vi in range(view_size):
                            if orig_batch is None:
                                new_images[vi][bi] = None
                            else:
                                orig = img[bi][vi]
                                new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig
        
                    dict_batch[img_key] = new_images
        
                txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        
                if len(txt_keys) != 0:
                    texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
                    encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
                    draw_text_len = len(encodings)
                    flatten_encodings = [e for encoding in encodings for e in encoding]

                    flatten_mlms = mlm_collator(flatten_encodings)
        
                    for i, txt_key in enumerate(txt_keys):
                        texts, encodings = (
                            [d[0] for d in dict_batch[txt_key]],
                            [d[1] for d in dict_batch[txt_key]],
                        )
        
                        mlm_ids, mlm_labels = (
                            flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                            flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                        )
        
                        input_ids = torch.zeros_like(mlm_ids)
                        attention_mask = torch.zeros_like(mlm_ids)
                        for _i, encoding in enumerate(encodings):
                            _input_ids, _attention_mask = (
                                torch.tensor(encoding["input_ids"]),
                                torch.tensor(encoding["attention_mask"]),
                            )
                            input_ids[_i, : len(_input_ids)] = _input_ids
                            attention_mask[_i, : len(_attention_mask)] = _attention_mask
        
                        dict_batch[txt_key] = texts
                        dict_batch[f"{txt_key}_ids"] = input_ids
                        dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                        dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                        dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                        dict_batch[f"{txt_key}_masks"] = attention_mask
            else:
                if "text_ids" in batch[0]:
                    data_to_collate = []
                    for item in batch:
                        collation_item = {}
                        collation_item["input_ids"] = item["text_ids"]
                        collation_item["attention_mask"] = item["text_ids"]
                        data_to_collate.append(collation_item)

                    collation_result = mlm_collator(data_to_collate)
                    dict_batch = {}
                    dict_batch["text_ids_mlm"] = collation_result["input_ids"]
                    dict_batch["text_masks"] = collation_result["attention_mask"]
                    dict_batch["text_labels_mlm"] = collation_result["labels"]
                    
        # Collate function for datasets derived from guggingface
        else:
            dict_batch = default_collate(batch)

        return dict_batch
