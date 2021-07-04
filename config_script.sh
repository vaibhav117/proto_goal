#! /bin/bash
# project_name=('cartpole_swingup_sparse' 'acrobot_swingup' 'cheetah_run' 'hopper_hop' 'reach_duplo' 'reacher_hard' 'walker_run' 'quadruped_run' 'hopper_stand' 'walker_stand' 'walker_walk' 'quadruped_walk' 'reacher_easy' 'cartpole_swingup'  'pendulum_swingup')
project_name=('point_mass_sb3')
echo ${project_name[$SLURM_ARRAY_TASK_ID]} 
python test_custom_env.py 