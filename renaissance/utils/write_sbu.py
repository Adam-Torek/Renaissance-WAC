import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
import copy

from tqdm import tqdm
from glob import glob
from PIL import Image

def write_sbu(root, dataset_root):
    captions_list = []
    image_paths_list = []
    for root, folders, _ in os.walk(root):
        for folder in folders:
                for file in os.listdir(os.path.join(root, folder)):   
                    if ".txt" in file:
                        captions_list.append(os.path.join(root, folder, file))
                    if ".jpg" in file:
                        image_paths_list.append(os.path.join(root, folder, file))
    
    captions_list.sort()
    image_paths_list.sort()

    column_values = []
    for i, image_path in enumerate(image_paths_list):
        caption = captions_list[i]
        image = open(image_path, "rb").read()
        column_values.append([caption, image, i])

    del image_paths_list, captions_list

    sbu_dataframe = pd.DataFrame(column_values, columns=["caption","image","image_id"])
    train_dataframe = sbu_dataframe.sample(frac=0.8)
    val_test_dataframe = copy.deepcopy(sbu_dataframe[~sbu_dataframe["image_id"].isin(train_dataframe["image_id"])])

    del sbu_dataframe

    test_dataframe = val_test_dataframe.sample(frac=0.5)
    val_dataframe = copy.deepcopy(val_test_dataframe[~val_test_dataframe["image_id"].isin(test_dataframe["image_id"])])

    train_dataframe["split"] = "train"
    val_dataframe["split"] = "val"
    test_dataframe["split"] = "test"

    table_dict = {
         "train": pa.Table.from_pandas(train_dataframe),
         "val": pa.Table.from_pandas(val_dataframe),
         "test": pa.Table.from_pandas(test_dataframe),
    }

    del train_dataframe, val_dataframe, test_dataframe

    os.makedirs(dataset_root, exist_ok=True)

    for subsplit, table in table_dict.items():
        with pa.OSFile(f"{dataset_root}/sbu_{subsplit}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as pa_file:
                pa_file.write_table(table)
    
    gc.collect()