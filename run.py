import os
import copy
import pytorch_lightning as pl
# import os
import sys
# os.environ["NCCL_DEBUG"] = "INFO"

from renaissance.config import ex
from renaissance.modules import RenaissanceTransformer
from renaissance.datamodules.multitask_datamodule import MTDataModule

from transformers import AutoConfig, AutoModel

from renaissance.modules.config import WACConfig

import warnings
import torch
import torch.distributed as dist

from renaissance.modules.two_tower_encoder import BertWACTransformer

def parse_load_path(load_path):
        drive, path_and_file = os.path.splitdrive(load_path)
        path, file = os.path.split(path_and_file)
        folders = []
        while True:
            path, folder = os.path.split(path)
        
            if folder != "":
                folders.append(folder)
            else:
                if path != "":
                    folders.append(path)
                break
        folders.reverse()
        result_dir = folders[-3]
        checkpoint_name = file.split("/")[-1][:-5]
        parsed_string = f"{result_dir}_{checkpoint_name}"
        return parsed_string

def run_experiment(_config):
     # print(_config)
    
    dm = MTDataModule(_config, dist=True)

    model = RenaissanceTransformer(_config)

    complete_encoder_path = _config["complete_encoder_path"]

    if complete_encoder_path is not None:
        local_directory = _config["huggingface_save_directory"]
        model.from_pretrained(complete_encoder_path, local_directory)
    
    # Create name for directory to log results
    load_path = _config['load_path']
    exp_name = f'{_config["exp_name"]}'
    seed = _config['seed']
    log_dir = _config['log_dir']
    
    print('\n\n')
    print('Running Renaissance vision-language platform with:', file=sys.stderr)
    print('Experiment Info')
    print('Experiment Name: ')
    print(f'Task: {exp_name}', file=sys.stderr)
    # print('Log Dir: ', model.logger.log_dir)
    print()
    print("Model Info")
    print("Model Type: ", _config['model_type'])
    if _config['model_type'] == 'one-tower':
        print("Encoder: ", _config['encoder'])
        print("Random Init: ", _config['random_init_encoder'])
        print("Manual Config: ", _config['encoder_manual_configuration'])
        print("Image Size: ", _config['image_size'])
        print("Patch Size: ", _config['patch_size'])
        
    elif _config['model_type'] == 'two-tower':
        print("Image Encoder: ", _config['image_encoder'])
        print("Freeze Image Encoder: ", _config['freeze_image_encoder'])
        print("Image Enc Random Init: ", _config['random_init_vision_encoder'])
        print("Image Enc Manual Config: ", _config['image_encoder_manual_configuration'])
        print("Image Size: ", _config['image_size'])
        print("Patch Size: ", _config['patch_size'])
        print("Text Encoder: ", _config['text_encoder'])
        print("Freeze Text Encoder: ", _config['freeze_text_encoder'])
        print("Text Enc Random Init: ", _config['random_init_text_encoder'])
        print("Text Enc Manual Config: ", _config['text_encoder_manual_configuration'])
        print("Max Text Langth: ", _config['max_text_len'])
        print("Vocab Size: ", _config['vocab_size'])
        
    print()
    print("Training Info")
    print("Data Root: ", _config['data_root'])
    print("Learning Rate: ", _config['learning_rate'])
    print("Max Epochs: ", _config['max_epoch'])
    print("Max Steps: ", _config['max_steps'])
    print("Warmup Steps: ", _config['warmup_steps'])
    print("LR Mult Head: ", _config['lr_mult_head'])  
    print("LR Mult Cross Modal: ", _config['lr_mult_cross_modal'])
    print('\n\n')

    print()
    print("WAC embeddings settings: ", _config["use_wac_embeddings"])
    print("Attention head part to inject WAC distributions: ", _config["wac_distribution_matrix"])
    print("Image encoder for WAC image features: ", _config["wac_image_encoder"])
    print("Save directory for WAC models", _config["wac_repo_id"])
    print("Num WAC Cores: ", _config["num_cores"])
    print("Local WAC Directory: ", _config["local_wac_directory"])

               
    if not load_path:
        image_size = _config['image_size']
        patch_size = _config['patch_size']
        batch_size = _config['batch_size']
        per_gpu_batchsize = _config['per_gpu_batchsize']
        train_steps = _config['max_steps']
        train_epoch = _config['max_epoch']
        result_dir = f"{exp_name}"
        for loss_name, loss_value in _config["loss_names"].items():
            if loss_value > 0:
                result_dir += f"_{loss_name}"

        result_dir += f"_seed{seed}_is{image_size}_ps{patch_size}_bs{batch_size}_pgbs{per_gpu_batchsize}_ts{train_steps}"
        if complete_encoder_path is not None:
            complete_encoder_path = complete_encoder_path.replace("/","_")
            result_dir += f"_{complete_encoder_path}"
    else:
        loaded_model = parse_load_path(load_path)
        result_dir = f"{exp_name}_seed{seed}_from_{loaded_model}"
        
    # Info Variables
    exp_name = _config['exp_name']
    
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=result_dir
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        devices= _config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator = 'gpu',
        strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        deterministic='warn',
        max_epochs=_config["max_epoch"], #if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        num_sanity_val_steps=0,
    )

    
    if _config["run_training"]:
        if model.wac_models is not None and _config["loss_names"]["ref"] > 0:
            print("WAC models enabled. Starting construction of WAC features.")
            dm.setup(stage="fit")
            training_dataloader = dm.train_dataloader()
            val_dataloader = dm.val_dataloader()

            model.build_wac_features(training_dataloader, split="train")
            model.build_wac_features(val_dataloader, split="val")

            model.delete_wac_image_encoder()

            print("WAC datasets constructed. Starting model training.")
        elif model.wac_models is not None and _config["loss_names"]["ref"] == 0:
            model.wac_models.training_completed = True

        if _config["resume_from"]:
            trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        else:
            trainer.fit(model, datamodule=dm)

        if _config["huggingface_save_directory"] is not None and _config["huggingface_save_name"] is not None:
            if "/" in _config["huggingface_save_name"]:
                model_save_name = _config["huggingface_save_name"].split("/")[-1]
            else:
                model_save_name = _config["huggingface_save_name"]
            
            model_save_directory = os.path.join(_config["huggingface_save_directory"], model_save_name)

            model.save_pretrained(model_save_directory, subsection_to_save=_config["subsection_to_save"])
            if _config["push_to_hub"]:
                model.push_to_hub(_config["huggingface_save_name"])
        
        # Display location of results
        print()
        print('Results can be found in:')
        print(model.logger.log_dir)
        print()
        
    if _config["run_test"]:
        if model.wac_models is not None:
            print("WAC models enabled. Starting construction of WAC features.")
            dm.setup(stage="test")
            #test_dataloader = dm.test_dataloader()
            
            #model.build_wac_features(test_dataloader, split="test")
            print("WAC datasets constructed. Starting model evaluation.")

        trainer.test(model, datamodule=dm)


@ex.automain
def main(_config):
    AutoConfig.register(model_type="bert_wac", 
                        config=WACConfig)
    
    AutoModel.register(config_class=WACConfig, 
                       model_class=BertWACTransformer)
    
    run_experiment(_config)

    