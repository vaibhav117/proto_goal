import os
os.environ['MUJOCO_GL'] = 'egl'
import torch
import copy
import numpy as np
from arguments import parse_args
import torch.nn as nn
import time
from datetime import datetime
from pytz import timezone
from MakeTreeDir import MAKETREEDIR
from video import VideoRecorder
from pathlib import Path
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from proto import make_agent
import fetch_image_env
# import pdb
torch.backends.cudnn.benchmark = True

def main(args):
    global ts
    now_asia = datetime.now(timezone("Asia/Kolkata"))
    format_day = "%m-%d"
    format_time = "%H:%M"

    ts_day= now_asia.strftime(format_day)
    ts_time = now_asia.strftime(format_time)
    args.work_dir = f"/scratch/sh6317/research/proto/exp_final/{args.env}/{ts_day}"

    exp_name =  args.env + '-' + ts_time + '-t' + str(args.num_train_steps) + '-s' + str(args.seed) 
    
    global directory
    directory = MAKETREEDIR()
    args.exp_dir = os.path.join(args.work_dir, exp_name)

    model_dir = os.path.join(args.exp_dir, "model")
    buffer_dir = os.path.join(args.exp_dir, "buffer")
    directory.makedir(args.work_dir)
    directory.makedir(args.exp_dir)
    directory.makedir(model_dir)
    directory.makedir(buffer_dir)

    L = Logger(args.exp_dir, save_tb=args.log_save_tb,
               log_frequency=args.log_frequency_step,
               action_repeat=args.action_repeat,
               agent="proto_rl")
    utils.set_seed_everywhere(args.seed)
    device = torch.device(args.device)
    env = fetch_image_env.make(args.env, args.frame_stack, args.action_repeat,args.max_episode_steps,
                            args.seed, fixed=args.fixed, reward_type=args.reward_type)
    # eval_env = fetch_image_env.make(args.env, args.frame_stack, args.action_repeat,args.max_episode_steps,
    #                         args.seed + 1, fixed=args.fixed, reward_type=args.reward_type)
    eval_env = env
    obs_spec = env.observation_space['image_observation']
    action_spec = env.action_space
    action_range = [float(action_spec.low.min()),float(action_spec.high.max())
        ]
    log_std_bounds =[-10, 2]
    expl_agent = make_agent(args,obs_spec.shape,action_spec.shape,action_range,device,log_std_bounds,task_agnostic=True)
    task_agent = make_agent(args,obs_spec.shape,action_spec.shape,action_range,device,log_std_bounds,task_agnostic=False) 
    expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape, # image_observation
                               args.replay_buffer_capacity,device)
    task_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,  # image_observation
                               args.replay_buffer_capacity,
                               device)
                               

    run(args,L,env,eval_env,expl_agent,task_agent,expl_buffer, task_buffer,model_dir)
def get_agent(args, step, expl_agent, task_agent):

    if step < args.num_expl_steps:
        return expl_agent
    return task_agent

def get_buffer(args,step,expl_buffer, task_buffer):
    if step < args.num_expl_steps:
        return expl_buffer
    return task_buffer   

def evaluate(args,L,eval_env, expl_agent,task_agent,step):
    avg_episode_reward = 0
    eval_video_recorder = VideoRecorder(args.exp_dir if args.save_video else None)
    for episode in range(args.num_eval_episodes):
        obs = eval_env.reset() 
        eval_video_recorder.init(enabled=(episode == 0))
        episode_reward = 0
        episode_success = 0
        episode_step = 0
        done= False
        while not done:
            agent = get_agent(args,step, expl_agent,task_agent)
            with utils.eval_mode(agent):
                obs= obs['image_observation']
                action = agent.act(obs, sample=False)
            obs, reward, done, info = eval_env.step(action)
            eval_video_recorder.record(eval_env)
            episode_reward += reward # reward
            episode_step += 1
        avg_episode_reward += episode_reward
        eval_video_recorder.save(f'{step}_fetch.mp4')
    avg_episode_reward /= args.num_eval_episodes

    L.log('eval/episode_reward', avg_episode_reward, step)
    L.dump(step, ty='eval')
def run(args,L,env,eval_env,expl_agent,task_agent,expl_buffer, task_buffer,model_dir):
    episode, episode_reward, episode_step = 0, 0, 0
    start_time = time.time()
    done = True
    step= 0 
    while step <= args.num_train_steps:
        if done:
            if step > 0:
                fps = episode_step / (time.time() - start_time)
                L.log('train/fps', fps,step)
                start_time = time.time()
                L.log('train/episode_reward', episode_reward, step)
                L.log('train/episode', episode, step)
                L.dump(step, ty='train')

            obs = env.reset()['image_observation']
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        agent = get_agent(args,step,expl_agent,task_agent)
        replay_buffer = get_buffer(args,step,expl_buffer,task_buffer)

        if step % args.eval_frequency == 0:
            L.log('eval/episode', episode - 1, step)
            evaluate(args,L,eval_env, expl_agent,task_agent,step)

            # save agent periodically
        if args.save_model and step % args.save_frequency == 0:
            utils.save(
                expl_agent,
                os.path.join(model_dir, f'expl_agent_{step}.pt'))
            utils.save(
                task_agent,
                os.path.join(model_dir, f'task_agent_{step}.pt'))
            # if self.cfg.save_buffer and step % self.cfg.save_frequency == 0:
                # replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)

            # sample action for data collection
        if step < args.num_random_steps:
            # spec = self.env.action_space.sample()
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        
        agent.update(replay_buffer, step)
        next_obs,reward,done, info = env.step(action)
        next_obs = next_obs['image_observation']

        # allow infinite bootstrap
        # done = time_step.last()
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_step += 1
        step += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
