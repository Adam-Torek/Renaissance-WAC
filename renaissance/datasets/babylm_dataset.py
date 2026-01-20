from transformers.data.data_collator import DataCollatorForLanguageModeling

from .base_dataset import BaseDataset


class BabyLMDataset(BaseDataset):

    def __init__(self, *args, split="", **kwargs) -> None:
        assert split in ["train", "val", "test"]
        self.split = split
        names = [f"babylm_{split}"]
        super().__init__(*args, **kwargs, names=names, text_column_name="sentences")

    def __len__(self) -> int:
        return len(self.table)
        
    def __getitem__(self, index) -> dict:
        text_string = self.table[self.text_column_name][index]
        text_string = str(text_string)

        encoding_output = self.tokenizer(text_string, 
                                         max_length=self.max_text_len, 
                                         padding="max_length",
                                         truncation=True,
                                         return_special_tokens_mask=True,
                                         return_tensors="pt")
        
        return_dict = {}
        return_dict["text_ids"] = encoding_output["input_ids"].squeeze()
        return_dict["text_mask"] = encoding_output["attention_mask"].squeeze()

        return return_dict
