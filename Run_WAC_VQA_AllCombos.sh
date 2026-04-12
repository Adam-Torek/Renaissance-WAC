#!/bin/bash

#SBATCH -J WAC_VQA_AllCombos         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_vqa_allcombos.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 128:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with pretrain_wac_enabled_vqa_twotower_electrasmall_deit_small \
                                    exp_name=pretrain_wac_enabled_embeddings_vqa_twotower_electrasmall_deit_small \
                                    wac_distribution_matrix=None \
                                    use_wac_embeddings=True

wac_distribution_matrices=(key query value)
wac_embedding_settings=(False True)

for wac_embedding_setting in "${wac_embedding_settings[@]}"
do
    for wac_distribution_matrix in "${wac_distribution_matrices[@]}"
    do

        if [[ "$wac_embedding_setting" == "True" ]]; then
            wac_embeddings=enabled_embeddings
        else
            wac_embeddings=disabled_embeddings
        fi

        srun python3 run.py with pretrain_wac_enabled_vqa_twotower_electrasmall_deit_small \
                                    exp_name=pretrain_wac_enabled_$wac_distribution_matrix\_$wac_embeddings\_vqa_twotower_electrasmall_deit_small \
                                    wac_distribution_matrix=$wac_distribution_matrix \
                                    use_wac_embeddings=$wac_embedding_setting

    done
done
