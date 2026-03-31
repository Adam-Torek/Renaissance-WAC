#!/bin/bash

#SBATCH -J Run_GLUE_DistEmbeds         # job name
#SBATCH -o slurm_logs/renaissance_wac/glue_distembeds.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 24:00:00                    # run time (hh:mm:ss)

source activate renaissance

glue_tasks=(cola mnli mrpc qqp qnli rte sst2 stsb wnli)
distribution_matrices=(key query value)

for distribution_matrix in "${distribution_matrices[@]}"
do
    for task in "${glue_tasks[@]}"
    do
        srun python3 run.py with finetune_$task\_twotower_electrasmall \
                                    complete_encoder_path=ajtorek/electra-deit-small-wac-$distribution_matrix\-distributions-embeddings \
                                    exp_name=finetune_twotower_electrasmall_wac_$distribution_matrix\_distributions_embeddings \
                                    wac_distribution_matrix=None \
                                    csv_log_file=glue_results/$distribution_matrix\_distembeds/glue.csv \
                                    use_wac_embeddings=True
                                    
    done 
done
