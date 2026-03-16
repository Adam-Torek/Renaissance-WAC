#!/bin/bash

#SBATCH -J WAC_Layers_Distributions_Eval         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_layer_distribution_eval.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 128:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with eval_ref_twotower_electrasmall_deit_wac_distributions \
                            wac_distribution_encoder_location=layers \
                            wac_distribution_encoder_sizes=[2048] \
                            huggingface_save_name=ajtorek/electra_deit_small_wac_layers_distributions \
                            max_epoch=1 \
                            exp_name=eval_ref_twotower_electrasmall_deit_wac_layers_distributions \
                           
