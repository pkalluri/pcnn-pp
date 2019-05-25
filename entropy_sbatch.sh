#!/bin/bash
#SBATCH --partition=jag-hi
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
#SBATCH --output=save/%j.out

# only use the following if you want email notification
# no SBATCH --mail-user=pkalluri@stanford.edu
# no SBATCH --mail-type=ALL

# list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

srun /sailhome/pkalluri/.conda/envs/pkalluri-pyp3.6.8/bin/python get_entropy.py -p save/_714767_samples/all_samples_from_params_cifar1.ckpt.npz -pd save/714767

# can try the following to list out which GPU you have access to
# srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

echo "Done"
