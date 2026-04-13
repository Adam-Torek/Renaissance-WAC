#!/bin/bash

#SBATCH -J Electra-DeIT-ITM      # job name
#SBATCH -o slurm_logs/renaissance_wac/electra_deit_itm.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=48      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:2
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 48:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 run.py with pretrain_wac_itm_twotower_electrasmall_deit_small
