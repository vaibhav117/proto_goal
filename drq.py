import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import kornia
import utils

def make_drq_agent(args,obs_shape, action_shape, action_range, device, log_std_bounds):
    return DRQAgent(obs_shape=obs_shape,
        action_shape=action_shape,action_range=action_range,device= device,
        discount=args.discount,init_temperature= args.init_temperature,
        lr=args.lr,actor_update_frequency=args.actor_update_frequency,
        critic_target_tau= args.critic_target_tau,critic_target_update_frequency=args.critic_target_update_frequency,
        batch_size=args.batch_size,num_seed_steps=args.num_seed_steps,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,hidden_depth=args.hidden_depth,log_std_bounds=log_std_bounds)



class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])






class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_shape, feature_dim,action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.encoder = Encoder(obs_shape=obs_shape,feature_dim=feature_dim)

        self.log_std_bounds = log_std_bounds

        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    # def log(self, logger, step):
    #     for k, v in self.outputs.items():
    #         logger.log_histogram(f'train_actor/{k}_hist', v, step)

    #     for i, m in enumerate(self.trunk):
    #         if type(m) == nn.Linear:
    #             logger.log_param(f'train_actor/fc{i}', m, step)




class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_shape,feature_dim, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = Encoder(obs_shape=obs_shape,feature_dim=feature_dim)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2





class DRQAgent(object):
    "not using proto encodings "

    def __init__(self, obs_shape, action_shape,feature_dim, action_range, device,hidden_dim,hidden_depth,
                log_std_bounds, discount, init_temperature, lr, actor_update_frequency, critic_target_tau,
                critic_target_update_frequency, batch_size,num_seed_steps):

        self.action_range= action_range
        self.device = device
        self.discount = discount
        self.critic_target_tau = critic_target_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.lr= lr
        self.num_seed_steps = num_seed_steps
        self.aug = nn.Sequential(nn.ReplicationPad2d(4), kornia.augmentation.RandomCrop((84, 84)))
        self.actor= Actor(obs_shape=obs_shape,feature_dim=feature_dim,action_shape=action_shape,
                          hidden_dim= hidden_dim, hidden_depth=hidden_depth,
                          log_std_bounds=log_std_bounds).to(self.device)

        self.critic = Critic(obs_shape=obs_shape, feature_dim=feature_dim, action_shape=action_shape,
                            hidden_dim=hidden_dim,hidden_depth=hidden_depth).to(self.device)




        self.critic_target= Critic(obs_shape=obs_shape, feature_dim=feature_dim, action_shape=action_shape,
                            hidden_dim=hidden_dim,hidden_depth=hidden_depth).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # init optimizers
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -action_shape[0]

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.train()
        self.critic_target.train()


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)


    @property
    def alpha(self):
        return self.log_alpha.exp()


    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])



    def update_critic(self, obs, action, reward, next_obs, discount, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                    target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)

            # dist_aug = self.actor(next_obs_aug)
            # next_action_aug = dist_aug.rsample()
            # log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
            #                                                         keepdim=True)
            # target_Q1, target_Q2 = self.critic_target(next_obs_aug,
            #                                             next_action_aug)
            # target_V = torch.min(
            #     target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            # target_Q_aug = reward + (not_done * self.discount * target_V)

            # target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Q1_aug, Q2_aug = self.critic(obs_aug, action)

        # critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
        #     Q2_aug, target_Q)

        # logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)


    def update_actor_and_alpha(self, obs, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # logger.log('train_actor/loss', actor_loss, step)
        # logger.log('train_actor/target_entropy', self.target_entropy, step)
        # logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        # logger.log('train_alpha/loss', alpha_loss, step)
        # logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()



    def update(self, replay_buffer, step):
        if len(replay_buffer) < self.num_seed_steps:
            return 
        obs, action, reward, next_obs, discount = replay_buffer.sample(self.batch_size,self.discount)
        
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)
        

        # logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs,action, reward, next_obs,discount,step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)






