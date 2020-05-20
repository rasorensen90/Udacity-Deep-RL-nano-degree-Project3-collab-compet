from utils import transpose_to_tensor, soft_update, hard_update
import torch
import torch.nn.functional as F
import numpy as np

class SelfPlayAgent:
    def __init__(self, config):
        self.config = config
        self.online_actor = config.actor_fn().to(self.config.device)
        self.target_actor = config.actor_fn().to(self.config.device)
        self.online_actor_opt = config.actor_opt_fn(self.online_actor.parameters())

        self.online_critic = config.critic_fn().to(self.config.device)
        self.target_critic = config.critic_fn().to(self.config.device)
        self.online_critic_opt = config.critic_opt_fn(self.online_critic.parameters())

        self.noises = [config.noise_fn() for _ in range(self.config.num_agents)]
        self.replay = config.replay_fn()

        hard_update(self.target_actor, self.online_actor) # initialize to be equal
        hard_update(self.target_critic, self.online_critic) # initialize to be equal
    
    

    def act(self, states):
        state = torch.from_numpy(states).float().to(self.config.device)

        self.online_actor.eval()

        with torch.no_grad():
            action = self.online_actor(state).cpu().numpy()

        self.online_actor.train()

        action += [n.sample() for n in self.noises]
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        full_state = states.flatten()
        next_full_state = next_states.flatten()
        self.replay.add((states, full_state, actions, rewards, next_states, next_full_state, dones))

        if len(self.replay) > self.replay.batch_size:
            self.learn()

    def learn(self):
        # Sample a batch from the replay buffer
        transitions = self.replay.sample()
        states, full_state, actions, rewards, next_states, next_full_state, dones = transpose_to_tensor(transitions, self.config.device)

        ### Update online critic model ###
        # Compute actions for next states with the target actor model
        with torch.no_grad(): # don't use gradients for target
            target_next_actions = [self.target_actor(next_states[:, i, :]) for i in range(self.config.num_agents)]

        target_next_actions = torch.cat(target_next_actions, dim=1)

        # Compute Q values for the next states and next actions with the target critic model
        with torch.no_grad(): # don't use gradients for target
            target_next_qs = self.target_critic(next_full_state.to(self.config.device), target_next_actions.to(self.config.device))

        # Compute Q values for the current states and actions
        target_qs = rewards.sum(1, keepdim=True) + self.config.discount * target_next_qs * (1 - dones.max(1, keepdim=True)[0])

        # Compute Q values for the current states and actions with the online critic model
        actions = actions.view(actions.shape[0], -1)
        online_qs = self.online_critic(full_state.to(self.config.device), actions.to(self.config.device))

        # Compute and minimize the online critic loss
        online_critic_loss = F.mse_loss(online_qs, target_qs.detach())
        self.online_critic_opt.zero_grad()
        online_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), 1)
        self.online_critic_opt.step()

        ### Update online actor model ###
        # Compute actions for the current states with the online actor model
        online_actions = [self.online_actor(states[:, i, :]) for i in range(self.config.num_agents)]
        online_actions = torch.cat(online_actions, dim=1)
        # Compute the online actor loss with the online critic model
        online_actor_loss = -self.online_critic(full_state.to(self.config.device), online_actions.to(self.config.device)).mean()
        # Minimize the online critic loss
        self.online_actor_opt.zero_grad()
        online_actor_loss.backward()
        self.online_actor_opt.step()

        ### Update target critic and actor models ###
        soft_update(self.target_actor, self.online_actor, self.config.target_mix)
        soft_update(self.target_critic, self.online_critic, self.config.target_mix)