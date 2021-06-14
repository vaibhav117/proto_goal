
import gym
import torch as th
from torch import nn
import utils
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int=256):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules  
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)

        obs_shape = observation_space['image_observation'].shape

     



        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Flatten(),)

        with th.no_grad():
            n_flatten = self.conv(
                th.as_tensor(observation_space.sample()["image_observation"]).float().unsqueeze(0)
            ).shape[1]

            # print(n_flatten,"------------------------- n-flatten")

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # self.repr_dim = 32 * 35 * 35  
        self.apply(utils.weight_init)

    def forward(self, observations, key) -> th.Tensor:
        # encoded_tensor_list = []

        # # self.extractors contain nn.Modules that do all the processing.
        # for key, extractor in self.extractors.items():
        #     encoded_tensor_list.append(extractor(observations[key]))
        # # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # return th.cat(encoded_tensor_list, dim=1)
        observations = observations[key]
        observations =observations / 255.
        h = self.linear(self.conv(observations))
        # h= h.view(h.shape[0], -1) 
        # print(h)
        return h

