from .base_dataset import BaseDataset
import torch


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train", "vqav2_val"]
        elif split == "val":
            names = ["vqav2_val"]
        elif split == "test":
            names = ["vqav2_test"]

        if "include_wac_data" in kwargs:
            include_wac_data = kwargs.pop("include_wac_data")
            self.include_wac_data = include_wac_data
        else:
            self.include_wac_data = False

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()
            
        return_dict =  {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }

        if self.include_wac_data:
            return_dict["tokenized_words"] = self.tokenizer.tokenize(text[0])
            return_dict["subimages"] = image_tensor[0]
            return_dict["position_data"] = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
            return_dict["ann_id"] = qid

        return return_dict
    
    def collate(self, batch, mlm_collator=None):
        batch_data = {}
        batch_data["image"] = [torch.stack([batch_item["image"][0] for batch_item in batch])]
        batch_data["text_ids"] = torch.stack([torch.tensor(batch_item["text"][1]["input_ids"]) for batch_item in batch])
        batch_data["text_masks"] = torch.stack([torch.tensor(batch_item["text"][1]["attention_mask"]) for batch_item in batch])
        batch_data["text_type_ids"] = torch.stack([torch.tensor(batch_item["text"][1]["token_type_ids"]) for batch_item in batch])
        batch_data["vqa_answer"] = [batch_item["vqa_answer"] for batch_item in batch]
        batch_data["vqa_labels"] = [batch_item["vqa_labels"] for batch_item in batch]
        batch_data["vqa_scores"] = [batch_item["vqa_scores"] for batch_item in batch]
        batch_data["qid"] = [batch_item["qid"] for batch_item in batch]
        
        if "tokenized_words" in batch[0]:
            batch_data["tokenized_words"] = [batch_item["tokenized_words"] for batch_item in batch]

        if "subimages" in batch[0]:
            batch_data["subimages"] = [torch.unsqueeze(batch_item["subimages"], 0) for batch_item in batch]
            batch_data["label"] = [0 for _ in range(0, len(batch))]

        if "position_data" in batch[0]:
            batch_data["position_data"] = [torch.unsqueeze(batch_item["position_data"], 0) for batch_item in batch]

        if "ann_id" in batch[0]:
            batch_data["ann_id"] = [batch_item["ann_id"] for batch_item in batch]


        return batch_data