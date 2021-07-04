#!/bin/bash 
#SBATCH --job-name=point_mass
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --array=0
#SBATCH --output=output/%j_%x_%N.out
#SBATCH --error=error/%j_%x_%N.err
#SBATCH --export=ALL

bash /scratch/sh6317/research/run-singularity.sh \
     /scratch/sh6317/research/proto_goal/config_script.sh
