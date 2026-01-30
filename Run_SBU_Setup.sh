#!/bin/bash

#SBATCH -J SBU_Dataset_Setup         # job name
#SBATCH -o slurm_logs/renaissance_wac/sbu_setup.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32      # Number of CPU nodes per task to run
#SBATCH -p bsudfq                    # queue (partition)
#SBATCH -t 24:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 renaissance/utils/run_write.py --dataset_to_write sbu --dataset_source_path data/sbucaptions --dataset_destination_path data/arrow/sbucaptions