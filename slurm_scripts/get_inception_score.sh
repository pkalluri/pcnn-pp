#!/bin/bash
#SBATCH --partition=atlas
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
#SBATCH --output=save/%j.out

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "CWD="$SLURM_SUBMIT_DIR

srun /u/nlp/anaconda/main/anaconda3/envs/ria-specialize/bin/python pcnn-pp/get_inception_score.py -p save/1774510_cifar_standard_samples_var0/5000_samples.npz

echo "Done"
