import argparse
from renaissance.modules import RenaissanceTransformer
import os

def main():
    upload_parser = argparse.ArgumentParser(prog="model_uploader.py", 
                                            description="Script to upload saved model checkpoint to HuggingFace." +
                                             "Note: This script will convert your model to the SafeTensors format before uploading it.")
    
    upload_parser.add_argument("--model_save_path", 
                               required=True, 
                               type=str, 
                               help="Path of the saved model to upload to HuggingFace.")
    
    upload_parser.add_argument("--model_upload_path", 
                               required=True, 
                               type=str, 
                               help="Path on HuggingFace Hub to upload model to online.")
    
    upload_parser.add_argument("--model_temp_save_path",
                               required=False,
                               type=str,
                               default="",
                               help="Temporary path to save converted model in SafeTensors format to."
                               )

    parsed_upload_arguments = upload_parser.parse_args()
    model_save_path = parsed_upload_arguments.model_save_path
    model_upload_path = parsed_upload_arguments.model_upload_path
    model_temp_save_path = parsed_upload_arguments.model_temp_save_path

    reinstantiated_model = RenaissanceTransformer.from_pretrained(model_save_path, local_files_only=True)
    if "/" in model_upload_path:
        model_save_name = model_upload_path.split("/")[-1]
    else:
        model_save_name = model_upload_path
    
    if model_temp_save_path == "":
        model_temp_save_path = os.getcwd()
    
    if not os.path.exists(model_temp_save_path):
        os.makedirs(model_save_path)

    reinstantiated_model.save_pretrained(os.path.join(model_save_path, model_save_name))
    os.remove(os.path.join(model_save_path, f"{model_save_name}.safetensors"))
    reinstantiated_model.push_to_hub(model_upload_path)

if __name__ == "__main__":
    main()