#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
#SBATCH --output=../save/%j.out



# only use the following if you want email notification
# no SBATCH --mail-user=pkalluri@stanford.edu
# no SBATCH --mail-type=ALL

# list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# can try the following to list out which GPU you have access to
# srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
echo "working directory = "$SLURM_SUBMIT_DIR

srun /u/nlp/anaconda/main/anaconda3/envs/ria-specialize/bin/python train_all_losses.py -g=1 -o=../save/$SLURM_JOB_ID -a=log_sum -d=cifar_sorted -c

echo "Done"
