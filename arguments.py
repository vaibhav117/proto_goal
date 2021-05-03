import argparse
import numpy as np
import wandb
import os 
def parse_args(s=None):

    if s is not None:
        parser = argparse.ArgumentParser(s)
    else:
        parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--env', default='fetch_reach')
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--action_repeat', default=2, type=int) # finger- 2 cartpole- 8 rest- 4
    parser.add_argument('--max_episode_steps', default=50, type=int)
    parser.add_argument('--reward_type', default='dense', type=str)
    parser.add_argument('--fixed', default=True,type=bool)

    # train
    parser.add_argument('--num_train_steps', default=500000, type=int)
    parser.add_argument('--num_expl_steps', default=250000, type=int)
    parser.add_argument('--num_random_steps', default=1000, type=int)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--seed', default=959, type=int)
    
    #eval 
    parser.add_argument('--eval_frequency', default=20000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)

    #misc
    parser.add_argument('--log_frequency_step', default=10000, type=int)
    parser.add_argument('--log_save_tb', default=True, type=bool)
    parser.add_argument('--save_video', default=True, type=bool)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--save_buffer', default=True, type=bool)
    parser.add_argument('--save_pixels', default=True, type=bool)
    parser.add_argument('--save_frequency', default=10000, type=int)
    parser.add_argument('--device', default="cuda", type=str)


    parser.add_argument('--load_pretrained', default=False, type=bool)
    parser.add_argument('--pretrained_step', default=250000, type=int)
    parser.add_argument('--pretrained_dir', default=None, type=str)


    # agent 
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--actor_update_frequency', default=2, type=int)
    parser.add_argument('--critic_target_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_frequency', default=2, type=int)
    parser.add_argument('--encoder_target_tau', default=0.05, type=float)
    parser.add_argument('--encoder_update_frequency', default=2, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--intr_coef', default=0.2, type=float)
    parser.add_argument('--num_seed_steps', default=1000, type=int)


    #critic
    parser.add_argument('--feature_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--hidden_depth', default=2, type=int)

    #encoder
    parser.add_argument("--proj_dim", default=128, type=int)

    # proto
    parser.add_argument("--pred_dim", default=512, type=int)
    parser.add_argument("--T", default=0.1, type=float)
    parser.add_argument("--num_protos",  default=512, type=int)
    parser.add_argument("--num_iters",  default=3, type=int)
    parser.add_argument("--topk",  default=3, type=int)
    parser.add_argument("--queue_size",  default=2048, type=int)



    args = parser.parse_args()
    return args




    