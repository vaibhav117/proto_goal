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

def make_agent(args,obs_shape,action_shape,action_range,device,log_std_bounds,task_agnostic, model_dir=None):

    

    if model_dir :
        
        chunks = os.listdir(model_dir)
        step = lambda x:int(x.split('.')[0].split('_')[-1])

        # if val is None:  
            
        val = max(step(x) for x in chunks)
        print("loading step :",val)
        a = torch.load('%s/expl_agent_%d.pt' % (model_dir, val))
        print(a.__dict__)


        # agent.load_state_dict(
        #     torch.load('%s/expl_agent_%d.pt' % (model_dir, val))
        # )
        return a

    else:
        agent= ProtoAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        action_range = action_range,
        device=device, discount=args.discount, 
        init_temperature=args.init_temperature, lr=args.lr,
        actor_update_frequency=args.actor_update_frequency,
        critic_target_tau=args.critic_target_tau, 
        critic_target_update_frequency=args.critic_target_update_frequency,
        batch_size=args.batch_size, intr_coef=args.intr_coef,num_seed_steps=args.num_seed_steps,
        proj_dim=args.proj_dim,feature_dim=args.feature_dim,hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth, log_std_bounds=log_std_bounds,
        pred_dim= args.pred_dim, num_protos=args.num_protos, topk=args.topk,T=args.T,
        queue_size=args.queue_size,encoder_target_tau=args.encoder_target_tau,
        encoder_update_frequency=args.encoder_update_frequency, task_agnostic=task_agnostic)

        return agent




def make_sac_agent(args,obs_shape, action_shape, action_range, device, log_std_bounds):
    return SacAgent(obs_shape=obs_shape,
        action_shape=action_shape,action_range=action_range,device= device,
        discount=args.discount,init_temperature= args.init_temperature,
        lr=args.lr,actor_update_frequency=args.actor_update_frequency,
        critic_target_tau= args.critic_target_tau,critic_target_update_frequency=args.critic_target_update_frequency,
        encoder_target_tau=args.encoder_target_tau,batch_size=args.batch_size,num_seed_steps=args.num_seed_steps,
        proj_dim=args.proj_dim,feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,hidden_depth=args.hidden_depth,log_std_bounds=log_std_bounds)

class Encoder(nn.Module):
    def __init__(self, obs_shape, proj_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.repr_dim = 32 * 35 * 35  # TODO check hard coded value

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
        self.pre_fc = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim))
        self.fc = utils.mlp(feature_dim, hidden_dim, 2 * action_shape[0],
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

        self.pre_fc = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim))
        self.Q1 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                            hidden_depth)
        self.Q2 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
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

class ProtoAgent(object):
    def __init__(self, obs_shape, action_shape, action_range, device,
                 discount,init_temperature, lr, actor_update_frequency,
                 critic_target_tau, critic_target_update_frequency,
                 encoder_target_tau, encoder_update_frequency, batch_size,
                 task_agnostic, intr_coef, num_seed_steps,proj_dim,feature_dim,hidden_dim,
                 hidden_depth, log_std_bounds,pred_dim,num_protos,topk,T,queue_size):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_tau = critic_target_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency
        self.batch_size = batch_size
        self.task_agnostic = task_agnostic
        self.intr_coef = intr_coef
        self.num_seed_steps = num_seed_steps
        self.lr = lr

        self.encoder = Encoder(obs_shape,proj_dim).to(self.device)
        self.encoder_target = Encoder(obs_shape,proj_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # actor_cfg.params.repr_dim = self.encoder.repr_dim
        # critic_cfg.params.repr_dim = self.encoder.repr_dim

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
        self.encoder.train(training)
        self.proto.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        obs = self.encoder.encode(obs)
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

        # get current Q estimates
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
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

    def update(self, replay_buffer, step):
        if len(replay_buffer) < self.num_seed_steps:
            return

        obs, action, extr_reward, next_obs, discount = replay_buffer.sample(
            self.batch_size, self.discount)

        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # train representation only during the task-agnostic phase
        if self.task_agnostic:
            if step % self.encoder_update_frequency == 0:
                self.update_repr(obs, next_obs, step)

                utils.soft_update_params(self.encoder, self.encoder_target,
                                         self.encoder_target_tau)

        with torch.no_grad():
            intr_reward = self.compute_reward(next_obs, step)

        if self.task_agnostic:
            reward = intr_reward
        else:
            reward = extr_reward + self.intr_coef * intr_reward

        # decouple representation
        with torch.no_grad():
            obs = self.encoder.encode(obs)
            next_obs = self.encoder.encode(next_obs)

        self.update_critic(obs, action, reward, next_obs, discount, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)



        
       

class SacAgent(object):
    """
    SAC with an auxiliary self-supervised task.
    Based on https://github.com/denisyarats/pytorch_sac_ae
    """
    def __init__(
        self,
        obs_shape,
        action_shape,action_range,device,discount,init_temperature,
        lr,actor_update_frequency,critic_target_tau,critic_target_update_frequency,
        encoder_target_tau,batch_size,num_seed_steps,
        proj_dim,feature_dim,hidden_dim,hidden_depth,log_std_bounds
      
        
    ):
        self.action_range= action_range
        self.device= device
        self.discount = discount
        self.critic_target_tau = critic_target_tau
        self.encoder_target_tau = encoder_target_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.num_seed_steps = num_seed_steps
        self.lr= lr
        self.batch_size= batch_size
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.aug = nn.Sequential(nn.ReplicationPad2d(4), kornia.augmentation.RandomCrop((84, 84)))
        self.encoder = Encoder(obs_shape,proj_dim).to(self.device)
        # self.encoder_target = Encoder(obs_shape,proj_dim).to(self.device)
        # self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.actor = Actor(self.encoder.repr_dim, feature_dim, action_shape, hidden_dim,
                 hidden_depth, log_std_bounds).to(self.device)

        self.critic = Critic(self.encoder.repr_dim,feature_dim,action_shape, hidden_dim,
                             hidden_depth).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim,feature_dim,action_shape, hidden_dim,
                             hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        


        # set target entropy to -|A|
        self.target_entropy = -action_shape[0] 
        # optimizers
        self.init_optimizers(lr)

        self.train()
        self.critic_target.train()
        # self.encoder_target.train()  TODO
        # actor_cfg.params.repr_dim = self.encoder.repr_dim
        # critic_cfg.params.repr_dim = self.encoder.repr_dim


        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        



        self.train()
        self.critic_target.train()
        # self.encoder_target.train()


    def init_optimizers(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=lr)
        
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def assign_modules_from(self, other):
        self.encoder = other.encoder
        # self.encoder_target = other.encoder_target
        # self.proto = other.proto
        self.actor = other.actor
        # init opts
        self.init_optimizers(self.lr)


        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder is not None:
            self.encoder.train(training)
       
    @property
    def alpha(self):
        return self.log_alpha.exp()



    def act(self, obs, sample=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            obs = self.encoder(obs, projection=False)
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
            target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)

        # get current Q estimates
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

  
        
    def update_actor_and_alpha(self, obs, step, update_alpha=True):
        # input encoded obs to actor
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs.detach(), action)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q.detach()).mean()


   


        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                        (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    

    
    def update(self, replay_buffer,step,L=None):
        if len(replay_buffer)< self.num_seed_steps:
            return
        
        obs, action, reward, next_obs, discount = replay_buffer.sample(self.batch_size,self.discount)
        
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)
        
        # r_mean= reward.mean()
        # if L is not None:
        #     L.log('train/batch_reward', r_mean, step)

        # not decouple encoder ? 
        # import pdb;pdb.set_trace()
        obs= self.encoder(obs, projection=False)
        next_obs = self.encoder(next_obs, projection=False)
        # print("before update critic ---------------------")
        self.update_critic(obs, action, reward, next_obs, discount, step=step)

        if step % self.actor_update_frequency == 0:
            # print("before update actor ******************")
            self.update_actor_and_alpha(obs, step=step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )

            # utils.soft_update_params(
            #     self.critic.Q2, self.critic_target.Q2, self.critic_tau
            # )
            # utils.soft_update_params(
            #     self.critic.encoder, self.critic_target.encoder,
            #     self.encoder_tau
            # )
        
        

    def save(self, model_dir, step):
        # loaded model will require gpu for inference 
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.state_dict(),  '%s/encoder_%s.pt' % (model_dir,step)
        )

    def load(self, model_dir,val=None):  # TODO later 
        print("loading ..")
        print(model_dir)
        chunks = os.listdir(model_dir)
        print(chunks)
        step = lambda x:int(x.split('.')[0].split('_')[-1])

        if val is None:  
            
            val = max(step(x) for x in chunks)
            print("loading step :",val)
            # chucks = [x for x in chunks if step(x)==val]
        #         buffer_file = chunks[-1]
        #         path = os.path.join(save_dir, buffer_file)
        #         payload = torch.load(path)
        assert self.actor is not None, "self.actor empty"
        assert self.critic  is not None, "self.critic empty"
        self.actor.load_state_dict(
            torch.load('%s/actor_%d.pt' % (model_dir, val))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%d.pt' % (model_dir, val))
        )
        self.encoder.load_state_dict(
            torch.load('%s/encoder_%d.pt' % (model_dir, val))
        )

        if self.ss_encoder is not None:
            self.ss_encoder.load_state_dict(
                torch.load('%s/ss_encoder_%d.pt' % (model_dir, val))
            )
            print("loaded ss encoder")
        
        # assert self.ss_encoder is not None, "self.ss_encoder empty"




