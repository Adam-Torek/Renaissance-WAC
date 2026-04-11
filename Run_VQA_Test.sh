#!/bin/bash

#SBATCH -J WAC_VQA_Test         # job name
#SBATCH -o slurm_logs/renaissance_wac/wac_vqa_test.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 128:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with pretrain_wac_vqa_twotower_electrasmall_deit_small
