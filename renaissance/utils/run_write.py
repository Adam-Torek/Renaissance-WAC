import argparse

from datasets import dataset_dict

from write_babylm import write_babylm
from write_refcoco import write_refcoco
from write_coco_karpathy import write_coco_karpathy
from write_conceptual_caption import write_conceptual_caption
from write_f30k_karpathy import write_f30k_karpathy
from write_nlvr2 import write_nlvr2
from write_sbu import write_sbu
from write_snli import write_snli
from write_vg import write_vg
from write_vqa import write_vqa

datasets_dict = {
    "babylm": write_babylm,
    "refcoco": write_refcoco,
    "coco_karpathy": write_coco_karpathy,
    "conceptual_caption": write_conceptual_caption,
    "f30k_karpathy": write_f30k_karpathy,
    "nlvr2": write_nlvr2,
    "sbu": write_sbu,
    "snli": write_snli,
    "vg": write_vg,
    "vqa" : write_vqa,
}

def main() -> None:
    write_processor = argparse.ArgumentParser(prog="run_write.py", 
                                              description="Program to process datasets into a normalized format for Renaissance.")
    
    write_processor.add_argument("--dataset_to_write", 
                                  type=str, 
                                  required=True, 
                                  choices=list(datasets_dict.keys()), 
                                  help="Dataset to write to a normalized arrow format")
    
    write_processor.add_argument("--dataset_source_path", 
                                  type=str, 
                                  required=True, 
                                  help="Source dataset path to convert to a normalized arrow format.")
    
    write_processor.add_argument("--dataset_destination_path", 
                                 type=str, 
                                 required=True, 
                                 help="Destination path to write normalized arrow format to.")
    
    write_dataset_args = write_processor.parse_args()

    dataset_to_write = write_dataset_args.dataset_to_write
    dataset_source_path = write_dataset_args.dataset_source_path
    dataset_destination_path = write_dataset_args.dataset_destination_path

    dataset_processing_function = datasets_dict[dataset_to_write]

    print(f"Starting to process dataset {dataset_to_write} from source path {dataset_source_path} to destination path {dataset_destination_path}")

    dataset_processing_function(dataset_source_path, dataset_destination_path)

    print(f"Finished processing dataset {dataset_to_write}")
    pass

if __name__ == "__main__":
    main()