#!/bin/bash

#SBATCH -J WAC_Distribution_Evaluation         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_distribution_eval.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 144:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with eval_ref_twotower_electrasmall_deit_wac_distributions
