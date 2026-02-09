#!/bin/bash

#SBATCH -J RefCOCO_Evaluations         # job name
#SBATCH -o slurm_logs/renaissance_wac/refcoco_evaluations.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:2
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 48:00:00                    # run time (hh:mm:ss)

source activate renaissance
complete_weights=("ajtorek/electra-deit-itm-renaissance-wac")

for weights in "${complete_weights[@]}"
do
    
    if [ -n "$weights" ]; then
        echo "Running experiments with weights $weights"
        srun python3 run.py with eval_ref_twotower_electrasmall_deit_small complete_encoder_path=$weights
    else
        echo "Running experiments without weights"
        srun python3 run.py with eval_ref_twotower_electrasmall_deit_small
    fi
done
