#!/bin/bash

#SBATCH -J WAC_Embeddings_Evaluation         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_embeddings_eval.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:2
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 144:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with eval_ref_twotower_electrasmall_deitsmall_wac_embeddings
