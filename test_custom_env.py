
import gym
print(__name__)
import numpy as np
import sys
from stable_baselines3 import  SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.her import   HerReplayBuffer
from stable_baselines3.her import goal_selection_strategy
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from fetch_image_env import  make_sb3_env
import gym
from stable_baselines3.common.env_checker import check_env
from gym.wrappers import Monitor
from video import VideoRecorder
from custom_td3 import TD3
from custom_policy import CustomTD3Policy, TD3Policy
from custom_feature_extractor import CustomCombinedExtractor
# env=make(env_name="fetch_reach", frame_stack=3, action_repeat=2, max_episode_steps=50, fixed=False, reward_type="dense")
from stable_baselines3.common.type_aliases import Schedule
from custom_her_replay_buffer import HerReplayBuffer
import pdb
pdb.set_trace()
# env = gym.make('HalfCheetah-v2')

# while True:
#     obs, r, done, info = env.step([0, 0, 0, 0, 0, 0])
#     if done: break


# env=make_sb3_env(env_name="fetch_reach", action_repeat=2, max_episode_steps=50, seed=10, fixed=False, reward_type="dense")
# env = Monitor(env, './video1', force=True)
# check_env(env)
# obs = env.reset()
# print(obs.keys())
# print(env.observation_space['image_observation'].shape)
# sys.exit()
# obs = env.reset()
# print(obs)
# env.reset()
# count=0
# v = VideoRecorder(video_dir="./video2")
# v.init(enabled=True)
# for i in range(100):
#     obs, reward, done, info = env.step(env.action_space.sample())
#     # env.render(mode="human")
#     v.record(env)
#     count+=1
#     if done:
#         print(count)
#         count=0
#         obs = env.reset()

# v.save('400_fetch.mp4')

goal_selection_strategy = "future"
online_sampling = True
max_episode_length = 50
env = make_sb3_env(env_name="fetch_reach", action_repeat=2, max_episode_steps=50, seed=10, fixed=False, reward_type="dense")

feature_extractor_class = CustomCombinedExtractor
feature_extractor_kwargs = [env.observation_space]

# custom_td3_policy= CustomTD3Policy(env.observation_space, env.action_space,
                                #    )
policy_kwargs = {
    
    "feature_extractor_class" : feature_extractor_class,
    "feature_extractor_kwargs" : feature_extractor_kwargs,
    "normalize_images": False
}

model = TD3(policy="CustomTD3Policy", env=env,learning_rate=1e-3,buffer_size=1000000,
            replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=max_episode_length,
            ),
            policy_kwargs=policy_kwargs,
            seed = 10,
            verbose=1,

             )

print(model)