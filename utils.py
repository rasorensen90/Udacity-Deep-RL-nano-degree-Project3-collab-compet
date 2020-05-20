import random
import torch
import copy
import numpy as np
from collections import deque

def transpose_to_tensor(tuples, device="cpu"):
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float).to(device)

    return list(map(to_tensor, zip(*tuples)))


def hard_update(target_model, source_model):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target_model, source_model, mix):
    for target_param, online_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - mix) + online_param.data * mix)

        
class Config:
    def __init__(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

        self.env = None
        self.brain_name = None
        self.num_agents = None
        self.state_size = None
        self.action_size = None

        self.actor_fn = None
        self.actor_opt_fn = None
        self.critic_fn = None
        self.critic_opt_fn = None
        self.replay_fn = None
        self.noise_fn = None
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.discount = None
        self.target_mix = None

        self.max_episodes = None
        self.max_steps = None

        self.actor_path = None
        self.critic_path = None
        self.scores_path = None

        
class Replay:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return random.sample(self.buffer, k=self.batch_size)

    def __len__(self):
        return len(self.buffer)
    
    
class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.state = None
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state
    
