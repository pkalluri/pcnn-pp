#!/bin/bash
#SBATCH --partition=atlas
#SBATCH --gres=gpu:1
#SBATCH --output=save/%j.out
# commenting out SBATCH --exclude=atlas13

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "CWD="$SLURM_SUBMIT_DIR

# command="/u/nlp/anaconda/main/anaconda3/envs/ria-specialize/bin/python pcnn-pp/data_setup/dataset2npz/omniglot2npz.py -d data"
command="/u/nlp/anaconda/main/anaconda3/envs/ria-specialize/bin/python code/train.py -d dbg -n 1 -q 1 -b 16 -o save/DBG -t 1"
echo $command
srun $command
echo "Done"