#!/bin/bash

#SBATCH -J GLUE-No-Positions-Eval       # job name
#SBATCH -o slurm_logs/renaissance_wac/no_positions_glue_eval.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 48:00:00                    # run time (hh:mm:ss)

source activate renaissance

glue_tasks=(cola mnli mrpc qqp qnli rte sst2 stsb wnli)

for task in "${glue_tasks[@]}"
do
    python3 run.py with finetune_$task\_twotower_electrasmall \
                                        complete_encoder_path=ajtorek/electra_deit_small_wac_embeddings_no_positions \
                                        use_position_data=False \
                                        csv_log_file=glue_results/glue_no_positions/glue.csv \
                                        exp_name=finetune_no_positions_$task\_twotower_electrasmall \

done