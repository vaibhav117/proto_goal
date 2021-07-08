
import utils

import gym
import numpy as np
import torch as th
import wandb
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy, BaseModel

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from torch.nn import functional as F

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.type_aliases import Schedule



def achieved_goal_from_obs(obs) -> str:
    temp=None
    if "achieved_goal_image" in obs.keys():
        temp = "achieved_goal_image"

    elif "image_observation" in obs.keys():
        temp="image_observation"

    return temp

# def create_mlp(2*features_dim, action_dim, net_arch, activation_fn, squash_output=True):

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """
    def __init__(self, observation_space,action_space, net_arch,
                 features_extractor, features_dim, activation_fn, normalize_images=False):
        
        super().__init__( observation_space,action_space, net_arch,
              features_extractor, features_dim, activation_fn, normalize_images)
        # Define custom network with Dropout
        # WARNING: it must end with a tanh activation to squash the output
        # keep normalize_images = False when calling function
        action_dim = get_action_dim(self.action_space)
        self.features_extractor= features_extractor
        self.actor_net = mlp([2*features_dim, *net_arch, action_dim],activation=activation_fn, output_activation=nn.Tanh)
        # actor_net = create_mlp(2*features_dim, action_dim, net_arch, activation_fn, squash_output=False)
        print(self.actor_net)
        self.apply(utils.weight_init)

    def forward(self,obs,no_grad_update=False): 
        # import pdb;pdb.set_trace()
        temp =achieved_goal_from_obs(obs)
        if no_grad_update:
            with th.no_grad():
                features1 = self.features_extractor(obs, temp)
                features2 = self.features_extractor(obs, "desired_goal_image")
                features = th.cat([features1,features2], dim=1)

        else:
            features1 = self.features_extractor(obs, temp)
            features2 = self.features_extractor(obs, "desired_goal_image")
            features = th.cat([features1,features2], dim=1)

        # TODO reformat above lines
        # print(features,"features actor------")
        action = self.actor_net(features)
        # print(action,f"---actor output {__file__}")
        wandb.log({"action_actor[0]": action.detach().cpu().numpy()[0][0],"action_actor[1]":  action.detach().cpu().numpy()[0][1] })
        return action


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = mlp([2*features_dim + action_dim, *net_arch, 1],activation=activation_fn)
            self.apply(utils.weight_init)
            # q_net = create_mlp(2*features_dim + action_dim, 1, net_arch, activation_fn)
            # q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)



    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            temp=achieved_goal_from_obs(obs)
            features_obs = self.features_extractor(obs,temp)
            features_goal = self.features_extractor(obs,"desired_goal_image")
            features = th.cat([features_obs, features_goal], dim=1)
            
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            temp=achieved_goal_from_obs(obs)
            features_achieved_goal = self.features_extractor(obs, temp)
            featured_desired_goal = self.features_extractor(obs, "desired_goal_image")
            features = th.cat([features_achieved_goal,featured_desired_goal], dim=1)
        return self.q_networks[0](th.cat([features, actions], dim=1))

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


    


register_policy("CustomTD3Policy", CustomTD3Policy)