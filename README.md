# GOAL Proto-RL: GOAL conditioned Reinforcement Learning with Prototypical Representations

This is a PyTorch implementation of **Goal conditioned Proto-RL** 





```

## Requirements
We assume you have access to a gpu that can run CUDA 11. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```
conda activate proto
```

## Instructions
1. Train proto-rl on the FetchEnv with fixed goal for num_expl_steps .Input the saved model dir for training goal conditioned RL 
2. Train Proto-rl aloong with a goal conditioned policy
```
python train.py env=fetch_reach  fixed=False
```
Note that we divede the number of steps by action repeat, which is set to 2 for all the environments.

This will produce the `exp_local` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```
tensorboard --logdir exp_local
```

Point mass env from https://github.com/jhejna/morphology-transfer/blob/main/bot_transfer/envs/assets/point_mass.xml 
