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
from proto import make_agent,make_sac_agent
from drq import make_drq_agent
from proto_goal_conditioned import GoalReplayBuffer,HERSampler,make_goal_agent
import fetch_image_env
# import pdb
torch.backends.cudnn.benchmark = True



## CODE CLeanup 



#Train proto_rl embeddings and run on downstream tasks 
# Keep fixed=True for fixed goal position
class ProtoTrain():
    def __init__(self,args):
        self.args =args
        now_asia = datetime.now(timezone("Asia/Kolkata"))
        format_day = "%m-%d"
        format_time = "%H:%M"

        ts_day= now_asia.strftime(format_day)
        ts_time = now_asia.strftime(format_time)
        self.args.work_dir = f"/scratch/sh6317/research/proto/exp_final/{self.args.env}/{ts_day}"

        exp_name = "PROTO_RL_" + args.env + '-' + ts_time + '-t' + str(args.num_train_steps) + '-s' + str(args.seed) 
    
        directory = MAKETREEDIR()
        self.args.exp_dir = os.path.join(args.work_dir, exp_name)

        self.model_dir = os.path.join(args.exp_dir, "model")
        self.buffer_dir = os.path.join(args.exp_dir, "buffer")
        directory.makedir(args.work_dir)
        directory.makedir(args.exp_dir)
        directory.makedir(self.model_dir)
        directory.makedir(self.buffer_dir)

        self.L = Logger(self.args.exp_dir, save_tb=args.log_save_tb,
                log_frequency=self.args.log_frequency_step,
                action_repeat=self.args.action_repeat,
                agent="proto_rl")
        utils.set_seed_everywhere(self.args.seed)
        self.device = torch.device(self.args.device)
        self.env = fetch_image_env.make(self.args.env, self.args.frame_stack, self.args.action_repeat,self.args.max_episode_steps,
                            self.args.seed, fixed=self.args.fixed, reward_type=self.args.reward_type)
        self.eval_env = self.env
        obs_spec = self.env.observation_space['image_observation']
        action_spec = self.env.action_space
        action_range = [float(action_spec.low.min()),float(action_spec.high.max())]
        log_std_bounds =[-10, 2]
        self.expl_agent = make_agent(self.args,obs_spec.shape,action_spec.shape,action_range,self.device,log_std_bounds,task_agnostic=True)
        self.task_agent = make_agent(self.args,obs_spec.shape,action_spec.shape,action_range,self.device,log_std_bounds,task_agnostic=False) 
        self.expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape, # image_observation
                               self.args.replay_buffer_capacity,self.device)
        self.task_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,  # image_observation
                               self.args.replay_buffer_capacity,
                               self.device)                   
    def get_agent(self, step):

        if step < self.args.num_expl_steps:
            return self.expl_agent
        return self.task_agent

    def get_buffer(self,step):
        if step < self.args.num_expl_steps:
            return self.expl_buffer
        return self.task_buffer   

    def evaluate(self,step):
        avg_episode_reward = 0
        eval_video_recorder = VideoRecorder(self.args.exp_dir if self.args.save_video else None)
        for episode in range(self.args.num_eval_episodes):
            obs = self.eval_env.reset() 
            eval_video_recorder.init(enabled=(episode == 0))
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            done= False
            while not done:
                agent = self.get_agent(step)
                with utils.eval_mode(agent):
                    obs= obs['image_observation']
                    action = agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_env.step(action)
                eval_video_recorder.record(self.eval_env)
                episode_reward += reward # reward
                episode_step += 1
            avg_episode_reward += episode_reward
            eval_video_recorder.save(f'{step}_fetch.mp4')
        avg_episode_reward /= self.args.num_eval_episodes

        self.L.log('eval/episode_reward', avg_episode_reward, step)
        self.L.dump(step, ty='eval')


    def run(self):
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        step= 0 
        while step <= self.args.num_train_steps:
            if done:
                if step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.L.log('train/fps', fps,step)
                    start_time = time.time()
                    self.L.log('train/episode_reward', episode_reward, step)
                    self.L.log('train/episode', episode, step)
                    self.L.dump(step, ty='train')

                obs = self.env.reset()['image_observation']
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.L.log('train/episode', episode, step)

            agent = self.get_agent(step)
            replay_buffer = self.get_buffer(step)

            if step % self.args.eval_frequency == 0:
                self.L.log('eval/episode', episode - 1, step)
                self.evaluate(step)

                # save agent periodically
            if self.args.save_model and step % self.args.save_frequency == 0:
                utils.save(
                    self.expl_agent,
                    os.path.join(self.model_dir, f'expl_agent_{step}.pt'))
                utils.save(
                    self.task_agent,
                    os.path.join(self.model_dir, f'task_agent_{step}.pt'))
                # if self.cfg.save_buffer and step % self.cfg.save_frequency == 0:
                    # replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)

                # sample action for data collection
            if step < self.args.num_random_steps:
                # spec = self.env.action_space.sample()
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=True)

            
            agent.update(replay_buffer, step)
            next_obs,reward,done, info = self.env.step(action)
            next_obs = next_obs['image_observation']

            # allow infinite bootstrap
            # done = time_step.last()
            episode_reward += reward

            replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_step += 1
            step += 1




## Train image based RL task with Proto training. This is based on DRQ 
class ScratchTrain():
    def __init__(self,args):
        self.args =args
        now_asia = datetime.now(timezone("Asia/Kolkata"))
        format_day = "%m-%d"
        format_time = "%H:%M"

        ts_day= now_asia.strftime(format_day)
        ts_time = now_asia.strftime(format_time)
        self.args.work_dir = f"/scratch/sh6317/research/proto/exp_final/{self.args.env}/{ts_day}"

        exp_name =  "Scratch-" + self.args.env + '-' + ts_time + '-t' + str(self.args.num_train_steps) + '-s' + str(self.args.seed) 
    
        directory = MAKETREEDIR()
        self.args.exp_dir = os.path.join(self.args.work_dir, exp_name)

        self.model_dir = os.path.join(self.args.exp_dir, "model")
        self.buffer_dir = os.path.join(self.args.exp_dir, "buffer")
        directory.makedir(self.args.work_dir)
        directory.makedir(self.args.exp_dir)
        directory.makedir(self.model_dir)
        directory.makedir(self.buffer_dir)

        self.L = Logger(self.args.exp_dir, save_tb=args.log_save_tb,
                log_frequency=self.args.log_frequency_step,
                action_repeat=self.args.action_repeat,
                agent="sac_rl")
        utils.set_seed_everywhere(self.args.seed)
        self.device = torch.device(self.args.device)
        self.env = fetch_image_env.make(self.args.env, self.args.frame_stack, self.args.action_repeat,self.args.max_episode_steps,
                            self.args.seed, fixed=self.args.fixed, reward_type=self.args.reward_type)
        self.eval_env = self.env
        obs_spec = self.env.observation_space['image_observation']
        action_spec = self.env.action_space
        action_range = [float(action_spec.low.min()),float(action_spec.high.max())]
        log_std_bounds =[-10, 2]
        self.task_agent = make_drq_agent(self.args,obs_spec.shape,action_spec.shape,action_range,self.device,log_std_bounds) 
        
        self.task_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,  # image_observation
                               self.args.replay_buffer_capacity,
                               self.device)                   
   

   
    def evaluate(self,step):
        avg_episode_reward = 0
        eval_video_recorder = VideoRecorder(os.path.join(self.args.exp_dir,"video") if self.args.save_video else None)
        for episode in range(self.args.num_eval_episodes):
            obs = self.eval_env.reset() 
            eval_video_recorder.init(enabled=(episode == 0))
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            done= False
            while not done:
                agent = self.task_agent
                with utils.eval_mode(agent):
                    obs= obs['image_observation']
                    action = agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_env.step(action)
                eval_video_recorder.record(self.eval_env)
                episode_reward += reward # reward
                episode_step += 1
            avg_episode_reward += episode_reward
            eval_video_recorder.save(f'{step}_fetch.mp4')
        avg_episode_reward /= self.args.num_eval_episodes

        self.L.log('eval/episode_reward', avg_episode_reward, step)
        self.L.dump(step, ty='eval')


    def run(self):
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        step= 0 
        while step <= self.args.num_train_steps:
            if done:
                if step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.L.log('train/fps', fps,step)
                    start_time = time.time()
                    self.L.log('train/episode_reward', episode_reward, step)
                    self.L.log('train/episode', episode, step)
                    self.L.dump(step, ty='train')

                obs = self.env.reset()['image_observation']
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.L.log('train/episode', episode, step)

            agent = self.task_agent
            replay_buffer = self.task_buffer

            if step % self.args.eval_frequency == 0:
                self.L.log('eval/episode', episode - 1, step)
                self.evaluate(step)

                # save agent periodically
            if self.args.save_model and step % self.args.save_frequency == 0:
            
                utils.save(
                    self.task_agent,
                    os.path.join(self.model_dir, f'task_agent_{step}.pt'))
                # if self.cfg.save_buffer and step % self.cfg.save_frequency == 0:
                    # replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)

                # sample action for data collection
            if step < self.args.num_random_steps:
                # spec = self.env.action_space.sample()
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=True)

            # print("before update ------------------------------------")
            
            agent.update(replay_buffer, step=step)
            next_obs,reward,done, info = self.env.step(action)
            next_obs = next_obs['image_observation']

            # allow infinite bootstrap
            # done = time_step.last()
            episode_reward += reward

            replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_step += 1
            step += 1



## Goal conditioned RL setting 
## 1. load pretrained proto embedding and use if for goal-conditioned RL training
## 2. Train a goal conditioned policy along with learning embeddings using proto RL 
class GoalRL():
    def __init__(self,args,load_pretrained_encoding=None):
        self.args =args
        now_asia = datetime.now(timezone("Asia/Kolkata"))
        format_day = "%m-%d"
        format_time = "%H:%M"
        
        ts_day= now_asia.strftime(format_day)
        ts_time = now_asia.strftime(format_time)
        self.args.work_dir = f"/scratch/sh6317/research/proto/exp_final2/{self.args.env}/{ts_day}"
        if load_pretrained_encoding:
            pre= 'GOAL-RL-PRETRAINED-07:19-s10'
        else:
            pre = 'GOAL-RL-PROTO_'
        exp_name =  pre+ self.args.env + '-' + ts_time + '-t' + str(self.args.num_train_steps) + '-s' + str(self.args.seed) 
    
        directory = MAKETREEDIR()
        self.args.exp_dir = os.path.join(self.args.work_dir, exp_name)

        self.model_dir = os.path.join(self.args.exp_dir, "model")
        self.buffer_dir = os.path.join(self.args.exp_dir, "buffer")
        directory.makedir(self.args.work_dir)
        directory.makedir(self.args.exp_dir)
        directory.makedir(self.model_dir)
        directory.makedir(self.buffer_dir)

        self.L = Logger(self.args.exp_dir, save_tb=args.log_save_tb,
                log_frequency=self.args.log_frequency_step,
                action_repeat=self.args.action_repeat,
                agent="proto_rl")
        utils.set_seed_everywhere(self.args.seed)
        self.device = torch.device(self.args.device)
        
        # for goal_conditioned rl , env reset changes goals #TODO verify
        self.env = fetch_image_env.make(self.args.env, self.args.frame_stack, self.args.action_repeat,self.args.max_episode_steps,
                            self.args.seed, fixed=self.args.fixed, reward_type=self.args.reward_type)
        self.eval_env = self.env
        obs_spec = self.env.observation_space['image_observation']
        action_spec = self.env.action_space
        action_range = [float(action_spec.low.min()),float(action_spec.high.max())]
        log_std_bounds =[-10, 2]
        if load_pretrained_encoding:
            self.expl_agent = make_agent(self.args,obs_spec.shape,action_spec.shape,action_range,self.device,log_std_bounds,task_agnostic=True, model_dir=load_pretrained_encoding)
            self.goal_agent = make_goal_agent(args,obs_spec.shape,action_spec.shape,action_range,self.device,log_std_bounds,pretrained_encoder=self.expl_agent.encoder)
        else:
            self.goal_agent = make_goal_agent(args,obs_spec.shape,action_spec.shape,action_range,self.device,log_std_bounds,pretrained_encoder=None)

        self.her = HERSampler(replay_strategy='future', replay_k=4,reward_func=self.goal_agent.compute_repr_dist_reward)
        # env_params


        env_params = {"max_env_steps": self.env._max_episode_steps,
                      "observation": self.env.observation_space['observation'].shape,
                      "achieved_goal" : self.env.observation_space['achieved_goal'].shape,
                      "achieved_goal_image":self.env.observation_space['image_observation'].shape,
                      "desired_goal":self.env.observation_space['desired_goal'].shape,
                      "desired_goal_image": self.env.observation_space['desired_goal_image'].shape,
                      "actions": action_spec.shape
                      }

        self.replay_buffer = GoalReplayBuffer(self.args.replay_buffer_capacity,self.her.sample_her_transitions,env_params)



    
    def run(self):
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        step= 0 
        while step <= self.args.num_train_steps:
            if done:
                if step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.L.log('train/fps', fps,step)
                    start_time = time.time()
                    self.L.log('train/episode_reward', episode_reward, step)
                    self.L.log('train/episode', episode, step)
                    self.L.dump(step, ty='train')

                obs = self.env.reset()
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.L.log('train/episode', episode, step)


            if step % self.args.eval_frequency == 0:
                self.L.log('eval/episode', episode - 1, step)
                self.evaluate(step)

                # save agent periodically
            if self.args.save_model and step % self.args.save_frequency == 0:
                
                utils.save(
                    self.goal_agent,
                    os.path.join(self.model_dir, f'goal_agent_{step}.pt'))
                # if self.cfg.save_buffer and step % self.cfg.save_frequency == 0:
                    # replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)

                # sample action for data collection
            if step < self.args.num_random_steps:
                # spec = self.env.action_space.sample()
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.goal_agent):
                    action = self.goal_agent.act(obs['image_observation'],obs['desired_goal_image'], sample=True)

            
            self.goal_agent.update(self.replay_buffer, step)
            next_obs,reward,done, info = self.env.step(action)

            # allow infinite bootstrap
            # done = time_step.last()
            episode_reward += reward

            self.replay_buffer.add(obs,next_obs, action)

            obs = next_obs
            episode_step += 1
            step += 1


    def evaluate(self,step):
        avg_episode_reward = 0
        eval_video_recorder = VideoRecorder(os.path.join(self.args.exp_dir,'video') if self.args.save_video else None)
        for episode in range(self.args.num_eval_episodes):
            obs = self.eval_env.reset() 
            eval_video_recorder.init(enabled=(episode == 0))
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            done= False
            while not done:
                
                with utils.eval_mode(self.goal_agent):
                    goal_image = obs['desired_goal_image']
                    obs= obs['image_observation']
                    action = self.goal_agent.act(obs, goal_image,sample=False)
                obs, reward, done, info = self.eval_env.step(action)
                eval_video_recorder.record(self.eval_env)
                episode_reward += reward # reward
                episode_step += 1
            avg_episode_reward += episode_reward
            eval_video_recorder.save(f'{step}_fetch.mp4')
        avg_episode_reward /= self.args.num_eval_episodes

        self.L.log('eval/episode_reward', avg_episode_reward, step)
        self.L.dump(step, ty='eval')



if __name__ == '__main__':
    args = parse_args()
    # model_dir= "/scratch/sh6317/research/proto/exp_final/fetch_reach/04-24/fetch_reach-04:34-t500000-s48/model/" 
    model_dir= "/scratch/sh6317/research/proto/exp_final/fetch_reach/05-19/PROTO_RL_fetch_reach-07:19-t500000-s10/model/"
    # update pre in init
    ob = GoalRL(args) 
    ob.run()
    # ob = ProtoTrain(args)
    # ob.run()


## learn embedding while doing the RL training (change in proto.py)
## use only the encoder from proto agent 