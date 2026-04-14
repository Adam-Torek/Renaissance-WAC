#!/bin/bash

#SBATCH -J WAC_VQA_EmbedDists         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_vqa_embeddists.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:2
#SBATCH -p shortgpu-l40                     # queue (partition)
#SBATCH -t 128:00:00                    # run time (hh:mm:ss)
#SBATCH --array=1-3

source activate renaissance

wac_distribution_matrices=(key query value)

for wac_distribution_matrix in "${wac_distribution_matrices[@]}"
do

    python3 run.py with pretrain_wac_enabled_vqa_twotower_electrasmall_deit_small \
                                exp_name=pretrain_wac_enabled_$wac_distribution_matrix\_embeddings_vqa_twotower_electrasmall_deit_small \
                                wac_distribution_matrix=$wac_distribution_matrix \
                                use_wac_embeddings=True \
                                num_gpus=2

done
