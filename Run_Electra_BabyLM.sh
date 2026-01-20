#!/bin/bash

#SBATCH -J Renaissance-Electra-BabyLM         # job name
#SBATCH -o slurm_logs/renaissance_wac/renaissance_electra_babylm.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 48:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with pretrain_mlm_onetower_electrasmall