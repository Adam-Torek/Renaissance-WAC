#!/bin/bash

#SBATCH -J GLUE_Base_Eval     # job name
#SBATCH -o slurm_logs/renaissance_wac/glue_base_eval.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 24:00:00                    # run time (hh:mm:ss)

source activate renaissance

for task in "${glue_tasks[@]}"
do
    srun python3 run.py with finetune_$task\_twotower_electrasmall \
                                use_wac_embeddings=False \
                                complete_encoder_path=ajtorek/electra-small-deit-small-ref \
                                csv_log_file=glue_results/glue_base_eval/glue.csv
                                
done 
