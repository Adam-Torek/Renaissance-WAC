#!/bin/bash

#SBATCH -J RefCOCO_Data_Preprocessing         # job name
#SBATCH -o slurm_logs/renaissance_wac/refcoco_data_preprocessing.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2      # Number of CPU nodes per task to run
#SBATCH -p bsudfq                   # queue (partition)
#SBATCH -t 24:00:00                    # run time (hh:mm:ss)

source activate renaissance

srun python3 renaissance/utils/run_write.py --dataset_to_write refcoco --dataset_source_path data/coco --dataset_destination_path data/arrow/coco
