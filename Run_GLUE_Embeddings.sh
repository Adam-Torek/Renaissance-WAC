#!/bin/bash

#SBATCH -J Embeddings_GLUE_Eval       # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_embeddings_glue.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 48:00:00                    # run time (hh:mm:ss)

source activate renaissance

glue_tasks=(cola mnli mrpc qqp qnli rte sst2 stsb wnli)
numtowers="two_tower"

for task in "${glue_tasks[@]}"
do
    srun python3 run.py with finetune_$task\_twotower_electrasmall \
                                use_wac_embeddings=True \
                                exp_name=finetune_$task\_twotower_electrasmall_embeddings \
                                csv_log_file=glue_results/glue_wac_embeddings/glue.csv \
                                wac_image_encoder=openai/clip-vit-base-patch16 \
                                complete_encoder_path=ajtorek/electra_deit_small_wac_embeddings

done 
