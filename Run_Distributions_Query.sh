#!/bin/bash

#SBATCH -J WAC_Query_Distributions_Eval         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_query_distribution_eval.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 128:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with eval_ref_twotower_electrasmall_deit_wac_distributions \
                            wac_distribution_encoder_location=all \
                            wac_distribution_matrix=query \
                            huggingface_save_name=ajtorek/electra_deit_small_wac_query_distributions \
                            push_to_hub=False \
                            max_epoch=1 \
                            exp_name=eval_ref_twotower_electrasmall_deit_wac_query_distributions
