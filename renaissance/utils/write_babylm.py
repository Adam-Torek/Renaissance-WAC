import glob
import os
import pyarrow as pa



def write_babylm(root, dataset_root) -> None:
    babylm_schema = pa.schema([("sentences", pa.string()), 
                           ("file_name", pa.string())])
    
    splits = ["10M", "100M","val","test"]

    for split in splits:
        if "10M" in split or "100M" in split:
            file_search_pattern = os.path.join(root, f"babylm_{split}", "*.train")
        elif "val" in split:
            file_search_pattern = os.path.join(root, "babylm_dev", "*.dev")
        else:
            file_search_pattern = os.path.join(root, "babylm_test", "*.test")

        file_names = glob.glob(file_search_pattern)
        split_dataset_dict = {"sentences": [], "file_name": []}
        for file_name in file_names:
            with open(file_name, "r") as babylm_text_file:
                for line in babylm_text_file:
                    split_dataset_dict["sentences"].append(line)
                    split_dataset_dict["file_name"].append(file_name)
        
        os.makedirs(dataset_root, exist_ok=True)
        outfile_path = os.path.join(dataset_root, f"babylm_{split}.arrow")
        
        pyarrow_table = pa.Table.from_pydict(split_dataset_dict)
        with pa.OSFile(outfile_path, "wb") as file_writer:
            with pa.RecordBatchFileWriter(file_writer, schema=babylm_schema) as batch_writer:
                batch_writer.write_table(pyarrow_table)
