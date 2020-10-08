#!/bin/bash
#SBATCH --partition=atlas
#SBATCH --gres=gpu:1
#SBATCH --output=save/%j.out
# commenting out SBATCH --exclude=atlas13

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "CWD="$SLURM_SUBMIT_DIR

command="/u/nlp/anaconda/main/anaconda3/envs/ria-specialize/bin/python code/quick_scripts/get_predictions.py save/1804440_cifar_standard_b16_l160_1810497_samples/100samples.npy save/1804440_cifar_standard_b16_l160_1810497_samples/100preds.npy"
echo $command
srun $command
echo "Done"