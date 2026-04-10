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
            return_dict["tokenized_text"] = self.tokenizer.tokenize(text)
            return_dict["subimages"] = image_tensor
            return_dict["position_data"] = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])

        return return_dict