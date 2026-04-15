#!/bin/bash

#SBATCH -J WAC_VQA_Embeds        # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_vqa_embeds.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:2
#SBATCH -p shortgpu-l40                     # queue (partition)
#SBATCH -t 128:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with pretrain_wac_enabled_vqa_twotower_electrasmall_deit_small \
                                    exp_name=pretrain_wac_enabled_embeddings_vqa_twotower_electrasmall_deit_small \
                                    wac_distribution_matrix=None \
                                    use_wac_embeddings=True \
                                    num_gpus=2
