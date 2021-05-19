# from _typeshed import OpenTextMode
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import copy
import math
import os
import utils
import kornia
from collections import defaultdict



def make_goal_agent(args,obs_shape,action_shape,action_range, device,log_std_bounds,pretrained_encoder=None):
    return ProtoGoalAgent(obs_shape=obs_shape, action_shape= action_shape,action_range= action_range, 
                 device = device, discount=args.discount,init_temperature=args.init_temperature,
                 lr =args.lr, actor_update_frequency = args.actor_update_frequency,
                 critic_target_tau= args.critic_target_tau, critic_target_update_frequency= args.critic_target_update_frequency,
                 encoder_target_tau=args.encoder_target_tau, encoder_update_frequency=args.encoder_update_frequency, 
                 batch_size = args.batch_size,intr_coef =args.intr_coef,num_seed_steps=args.num_seed_steps,
                 proj_dim= args.proj_dim,feature_dim=args.feature_dim,hidden_dim= args.hidden_dim,
                 hidden_depth=args.hidden_depth, log_std_bounds=log_std_bounds,pred_dim=args.pred_dim,
                 num_protos =args.num_protos,topk=args.topk,T=args.T,queue_size=args.queue_size,
                 pretrained_encoder=pretrained_encoder)


class Encoder(nn.Module):
    def __init__(self, obs_shape, proj_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.repr_dim = 32 * 35 * 35  

        self.projector = nn.Linear(self.repr_dim, proj_dim)

        self.apply(utils.weight_init)

    def encode(self, obs):
        obs = obs / 255.
        h = self.conv(obs)
        h = h.view(h.shape[0], -1)
        return h

    def forward(self, obs,projection=True):
        h = self.encode(obs)
        if projection:
            z = self.projector(h) #TODO remove projection for sacagent
            return z
        else:
            return h




class Actor(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_shape, hidden_dim,
                 hidden_depth, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.pre_fc = nn.Sequential(nn.Linear(2*repr_dim, 2*feature_dim),
                                    nn.LayerNorm(2*feature_dim))
        self.fc = utils.mlp(2*feature_dim, hidden_dim, 2 * action_shape[0],
                            hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.pre_fc(obs)
        mu, log_std = self.fc(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_shape, hidden_dim,
                 hidden_depth):
        super().__init__()

        self.pre_fc = nn.Sequential(nn.Linear(2*repr_dim, 2*feature_dim),
                                    nn.LayerNorm(2*feature_dim))
        self.Q1 = utils.mlp(2*feature_dim + action_shape[0], hidden_dim, 1,
                            hidden_depth)
        self.Q2 = utils.mlp(2*feature_dim + action_shape[0], hidden_dim, 1,
                            hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        h = self.pre_fc(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class Proto(nn.Module):
    def __init__(self, proj_dim, pred_dim, T, num_protos, topk,
                 queue_size):
        super().__init__()

        self.predictor = nn.Sequential(nn.Linear(proj_dim,
                                                 pred_dim), nn.ReLU(),
                                       nn.Linear(pred_dim, proj_dim))

        self.num_iters = 3
        self.T = T
        self.topk = topk
        self.num_protos = num_protos

        self.protos = nn.Linear(proj_dim, num_protos, bias=False)
        # candidate queue
        self.register_buffer('queue', torch.zeros(queue_size, proj_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, s, t):
        # normalize prototypes
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        s = self.predictor(s)

        s = F.normalize(s, dim=1, p=2)
        t = F.normalize(t, dim=1, p=2)

        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.T, dim=1)

        with torch.no_grad():
            scores_t = self.protos(t)
            q_t = self.sinkhorn(scores_t)

        loss = -(q_t * log_p_s).sum(dim=1).mean()
        return loss

    def compute_reward(self, z):
        B = z.shape[0]
        Q = self.queue.shape[0]
        assert Q % self.num_protos == 0

        # normalize
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        z = F.normalize(z, dim=1, p=2)

        scores = self.protos(z).T
        p = F.softmax(scores, dim=1)
        idx = pyd.Categorical(p).sample()

        # enqueue
        ptr = int(self.queue_ptr[0])
        self.queue[ptr:ptr + self.num_protos] = z[idx]
        self.queue_ptr[0] = (ptr + self.num_protos) % Q

        # compute distances
        z_to_q = torch.norm(z[:, None, :] - self.queue[None, :, :], dim=2, p=2)
        d, _ = torch.topk(z_to_q, self.topk, dim=1, largest=False)
        reward = d[:, -1:]
        return reward

    def sinkhorn(self, scores):
        def remove_infs(x):
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores / self.T
        Q -= Q.max()

        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for it in range(self.num_iters):
            u = Q.sum(dim=1)
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T



class ProtoGoalAgent(object):
    def __init__(self, obs_shape, action_shape, action_range, device,
                 discount,init_temperature, lr, actor_update_frequency,
                 critic_target_tau, critic_target_update_frequency,
                 encoder_target_tau, encoder_update_frequency, batch_size,
                 intr_coef, num_seed_steps,proj_dim,feature_dim,hidden_dim,
                 hidden_depth, log_std_bounds,pred_dim,num_protos,topk,T,queue_size, pretrained_encoder=None):


        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_tau = critic_target_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency
        self.batch_size = batch_size
        self.intr_coef = intr_coef
        self.num_seed_steps = num_seed_steps
        self.lr = lr
        self.pretrained_encoder= pretrained_encoder
        self.encoder = Encoder(obs_shape,proj_dim).to(self.device)
        if pretrained_encoder:
            self.encoder.load_state_dict(pretrained_encoder.state_dict())
        self.encoder_target = Encoder(obs_shape,proj_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())


        self.actor = Actor(self.encoder.repr_dim, feature_dim, action_shape, hidden_dim,
                 hidden_depth, log_std_bounds).to(self.device)

        self.critic = Critic(self.encoder.repr_dim,feature_dim,action_shape, hidden_dim,
                             hidden_depth).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim,feature_dim,action_shape, hidden_dim,
                             hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.proto = Proto(proj_dim, pred_dim, T, num_protos, topk,
                 queue_size).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        self.aug = nn.Sequential(nn.ReplicationPad2d(4),
                                 kornia.augmentation.RandomCrop((84, 84)))
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]
        # optimizers
        self.init_optimizers(lr)

        self.train()
        self.critic_target.train()
        if self.pretrained_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()
            self.encoder_target.train()

    def init_optimizers(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.proto_optimizer = torch.optim.Adam(utils.chain(
            self.encoder.parameters(), self.proto.parameters()),
                                                lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def assign_modules_from(self, other):
        self.encoder = other.encoder
        self.encoder_target = other.encoder_target
        self.proto = other.proto
        self.actor = other.actor
        # init opts
        self.init_optimizers(self.lr)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if not self.pretrained_encoder:
            self.encoder.train(training)
            self.proto.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs,goal, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        goal = torch.FloatTensor(goal).to(self.device)
        goal= goal.unsqueeze(0)
        # print(goal.shape, obs.shape)
        obs = self.encoder.encode(obs)
        goal = self.encoder.encode(goal)
        obs_goal = torch.cat((obs,goal),dim=1)
        dist = self.actor(obs_goal)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs_goal, action, reward, next_obs_goal, discounts,step):
        with torch.no_grad():
            dist = self.actor(next_obs_goal)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_goal, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discounts * target_V)

        # get current Q estimates
        Q1, Q2 = self.critic(obs_goal, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs_goal, step):
        dist = self.actor(obs_goal) # obs = repr_dim 
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs_goal, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_repr(self, obs, next_obs, step):
        z = self.encoder(obs)
        with torch.no_grad():
            next_z = self.encoder_target(next_obs)

        loss = self.proto(z, next_z)
        self.proto_optimizer.zero_grad()
        loss.backward()
        self.proto_optimizer.step()

    def compute_reward(self, next_obs, step):
        with torch.no_grad():
            y = self.encoder(next_obs)
            reward = self.proto.compute_reward(y)
        return reward
    def compute_repr_dist_reward(self,o,g,image_based=True):
        with torch.no_grad():
            if image_based:
                y_o= self.encoder.encode(o)
                y_g= self.encoder.encode(g)
                ## L2 norm
                # return -torch.norm(y_o-y_g,2)
                ## or use cosine similarity 
                return F.CosineSimilarity(y_o,y_g,dim=1) 
                # TODO check dimension 
                # TODO compute rewards from the projection

            


    def update(self, goal_replay_buffer, step):
        if len(goal_replay_buffer) < self.num_seed_steps:
            return

        transitions = goal_replay_buffer.sample(self.batch_size)
        discounts = np.ones(self.batch_size,1,dtype=np.float32)* self.discount
        # obs = self.aug(transitions[''])
        # next_obs = self.aug(next_obs)
        achieved_goal_image = transitions['achieved_goal']
        desired_goal_image = transitions['desired_goal_image']
        achieved_goal_image_next = transitions['achieved_goal_image_next']
        action = transitions['actions']
        achieved_goal_image = self.aug(achieved_goal_image)
        desired_goal_image = self.aug(desired_goal_image)
        achieved_goal_image_next = self.aug(achieved_goal_image_next)

        reward= transitions['reward']

        # train representation only during the task-agnostic phase
        # if self.task_agnostic:
        #     if step % self.encoder_update_frequency == 0:
        #         self.update_repr(obs, next_obs, step)

        #         utils.soft_update_params(self.encoder, self.encoder_target,
        #                                  self.encoder_target_tau)

        # with torch.no_grad():
        #     intr_reward = self.compute_reward(next_obs, step)

        # if self.task_agnostic:
        #     reward = intr_reward
        # else:
        # reward = extr_reward + self.intr_coef * intr_reward
        if not self.pretrained_encoder:
            if step % self.encoder_update_frequency == 0:
                self.update_repr(achieved_goal_image, achieved_goal_image_next, step)
                utils.soft_update_params(self.encoder, self.encoder_target,self.encoder_target_tau)
            with torch.no_grad():
                intr_reward = self.compute_reward(achieved_goal_image_next, step)
                reward = transitions['reward'] + self.intr_coef * intr_reward

        # decouple representation
        with torch.no_grad():
            achieved_goal_image = self.encoder.encode(achieved_goal_image)
            desired_goal_image = self.encoder.encode(desired_goal_image)
            achieved_goal_image_next = self.encoder.encode(achieved_goal_image_next)
            obs_goal = torch.cat((achieved_goal_image,desired_goal_image), dim=1)
            next_obs_goal= torch.cat((achieved_goal_image_next,desired_goal_image),dim=1)

        self.update_critic(obs_goal, action, reward, next_obs_goal, discounts, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs_goal, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)



class GoalReplayBuffer:

    # check this out 
    # https://github.com/vitchyr/rlkit/blob/v0.1.2/rlkit/data_management/obs_dict_replay_buffer.py

    def __init__(self, buffer_size, sample_func, env_params):
        self.env_params= env_params
        self.T= env_params["max_env_steps"]
        self.size= buffer_size // self.T
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # see observation_spec for dimensions
        self.episode = defaultdict(list)
        self.buffers = {'observation': np.empty([self.size, self.T , *self.env_params['observation']], dtype=np.float32),
                'achieved_goal': np.empty([self.size, self.T, *self.env_params['achieved_goal']], dtype=np.float32),
                'achieved_goal_image':np.empty([self.size, self.T , *self.env_params['achieved_goal_image']], dtype=np.uint8),
                'achieved_goal_image_next':np.empty([self.size, self.T , *self.env_params['achieved_goal_image']], dtype=np.uint8),
                'desired_goal': np.empty([self.size, self.T, *self.env_params['desired_goal']],dtype=np.float32),
                'desired_goal_image':np.empty([self.size, self.T, *self.env_params['desired_goal_image']],dtype=np.uint8),
                'actions': np.empty([self.size, self.T, *self.env_params['actions']], dtype=np.float32)
                
                }

    def sample(self, batch_size):
        temp_buffers={}
        for key in self.buffers.keys():
            temp_buffers[key]=self.buffers[key][:self.current_size]
        # temp_buffers['achieved_goal_next'] = temp_buffers['achieved_goal'][:, 1:, :]
        # temp_buffers['achieved_goal_image_next']=temp_buffers['achieved_goal_image'][:, 1:, :]

        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        
        # discounts = np.ones((batch_size, 1), dtype=np.float32) * discount
        # discounts = torch.as_tensor(discounts, device=self.device)
        return transitions

    def __len__(self):
        return self.current_size


    def add(self,obs,next_obs,action):
        self.episode['observation'].append(np.array(obs['image_observation']))
        self.episode['achieved_goal'].append(np.array(obs['achieved_goal']))
        self.episode['achieved_goal_image'].append(np.array(obs['image_observation']))

        self.episode['achieved_goal_image_next'].append(np.array(next_obs['image_observation']))
        self.episode['desired_goal'].append(np.array(obs['desired_goal']))
        self.episode['desired_goal_image'].append(np.array(obs['desired_goal_image']))
        self.episode['action'].append(np.array(action))
        
        if len(self.episode) ==  self.T:
            self.store_episode([np.array(self.episode['observation']),self.episode['achieved_goal'],
                                self.episode['achieved_goal_image_next'],self.episode['desired_goal'],
                                self.episode['desired_goal_image'],self.episode['action']])
            self.episode = defaultdict(list)

    def store_episode(self, episode_batch):
        observation,achieved_goal, achieved_goal_image, desired_goal,desired_goal_image,actions = episode_batch
        batch_size = observation.shape[0]
        # with self.lock:
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['achieved_goal'][idxs] = achieved_goal
        self.buffers['achieved_goal_image'][idxs] = achieved_goal_image
        self.buffers['achieved_goal_image_next'][idxs] = achieved_goal_image
        self.buffers['desired_goal'][idxs] = desired_goal
        self.buffers['desired_goal_image'][idxs] = desired_goal_image
        self.buffers['actions'][idxs] = actions
        self.n_transitions_stored += self.T * batch_size

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx



## HER Sampler 
class HERSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy= replay_strategy
        self.replay_k= replay_k
        # self.image_encoder= image_encoder 
        
        if self.replay_strategy == 'future':
            self.future_p= 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0

        self.reward_func = reward_func



     
    
    
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions ):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size= episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # print(rollout_batch_size,batch_size,T)
        # np.random.randint - low-inclusive and high-exclusive, hence max_env_steps+1 
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
          # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # TODO try future strategy with only last state? 

        # TODO current strategy randomly samples index from the her_index till last 
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)

        
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag_image = episode_batch['achieved_goal_image'][episode_idxs[her_indexes], future_t]
        transitions['desired_goal_image'][her_indexes] = future_ag_image
        # to get the params to re-compute reward 
        # image based dense reward compute 
        transitions['reward'] = np.expand_dims(self.reward_func(transitions['achieved_goal_image_next'], transitions['desired_goal_image'],image_based=True), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions



