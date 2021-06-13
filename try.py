

# https://github.com/DLR-RM/stable-baselines3/blob/88e1be9ff5e04b7688efa44951f845b7daf5717f/stable_baselines3/common/torch_layers.py 

# network architecture s
# since env is not goal.env hence custom policy is rewquired 

import gym
from gym.envs.robotics import utils
import highway_env
import numpy as np

from stable_baselines3 import  SAC, DDPG, TD3
import stable_baselines3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her import   HerReplayBuffer
from stable_baselines3.td3.policies import MultiInputPolicy
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from fetch_image_env import FetchReachImageEnv
import fetch_image_env
env = gym.make("parking-v0")




goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = 100

# # Initialize the model
model = CustomTD3(
    "custompolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
)





import gym
import torch as th
from torch import nn
import utils
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules  
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # extractors = {}

        # total_concat_size = 0
        # # We need to know size of the output of this extractor,
        # # so go over all the spaces and compute output feature sizes
        # for key, subspace in observation_space.spaces.items():
        #     if key == "image":
        #         # We will just downsample one channel of the image by 4x4 and flatten.
        #         # Assume the image is single-channel (subspace.shape[0] == 0)
        #         extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
        #         total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
        #     elif key == "vector":
        #         # Run through a simple MLP
        #         extractors[key] = nn.Linear(subspace.shape[0], 16)
        #         total_concat_size += 16
        obs_shape = observation_space['image_observation'].shape


        # self.extractors = nn.ModuleDict(extractors)
    
        # # Update the features dim manually
        # self._features_dim = total_concat_size




        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.repr_dim = 32 * 35 * 35   # TODO corrct this 
        self.apply(utils.weight_init)

    def forward(self, observations) -> th.Tensor:
        # encoded_tensor_list = []

        # # self.extractors contain nn.Modules that do all the processing.
        # for key, extractor in self.extractors.items():
        #     encoded_tensor_list.append(extractor(observations[key]))
        # # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # return th.cat(encoded_tensor_list, dim=1)

        observations =observations / 255.
        h = self.conv(observations)
        h= h.view(h.shape[0], -1) 
        return h



class CustomTD3(object):
    pass



# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )



## THe customcombinedFeature extractor is a input to the MultiInputPolicy in TD3 
### we dont need to modify the actor critic networks from inside 

### For off policy algorithms 

### Compared to their on-policy counterparts, no shared layers (other than the feature extractor) between the actor and the critic are allowed (to prevent issues with target networks).


## TODO make 2 versions --- 1. td3 without frame stacking 2.td3 with frame-stacking 3. Provision for proto rl feature extractor w/o frame stacking 


## TODO 

