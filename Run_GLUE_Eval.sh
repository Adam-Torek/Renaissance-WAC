#!/bin/bash

#SBATCH -J Renaissance-GLUE-Eval       # job name
#SBATCH -o slurm_logs/renaissance_wac/renaissance_electra_glue.o%j               # output and error file name (%j expands to jobID)  
#SBATCH --nodes=1			               # Number of nodes to run on
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      # Number of CPU nodes per task to run
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu-l40                     # queue (partition)
#SBATCH -t 48:00:00                    # run time (hh:mm:ss)

source activate renaissance

glue_tasks=(cola mnli mrpc qqp qnli rte sst2 stsb wnli)
numtowers=""

if [ "$1" = "one-tower" ]; then
    numtowers="onetower"
elif [ "$1" = "two-tower" ]; then
    numtowers="twotower"
else
    echo "$1 must be one-tower or two-tower"
    exit 1
fi

for task in "${glue_tasks[@]}"
do
    python3 run.py with finetune_$task\_$numtowers\_electrasmall
done 
