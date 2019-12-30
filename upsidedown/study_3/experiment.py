import numpy as np
import torch
from scipy.stats import median_absolute_deviation
import os
import torch.nn.functional as F
import time 
import random
import torch.nn as nn
           
def get_action(env, model, inputs, sample_action, epsilon):
    action_logits = model([inputs[:, :-2], inputs[:, -2:]])
    action_probs = torch.softmax(action_logits, axis=-1)

    if random.random() < epsilon: # Random action
        return env.action_space.sample()
    
    if sample_action:        
        m = torch.distributions.categorical.Categorical(logits=action_logits)             
        action = int(m.sample().squeeze().cpu().numpy())        
    else:
        action = int(np.argmax(action_probs.detach().squeeze().numpy()))
    return action

def rollout_episode(env, model, sample_action, cmd, 
                    render, device, epsilon, max_return=300):
    s = env.reset()
    done = False
    ep_reward = 0.0
    
    t = Trajectory()
    
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:            
            (dh, dr) = cmd
                
            inputs = torch.tensor([to_training(s, dr, dh)]).float().to(device)
            with torch.no_grad():
                model.eval()
                action = get_action(env, model, inputs, sample_action, epsilon=epsilon)
                model.train()
        
        if render:
            env.render()
            time.sleep(0.01)
            
        s_old = s        
        s, reward, done, info = env.step(action)
        
        if model is not None:
            dh = max(dh - 1, 1)
            dr = min(dr - reward, max_return)
            cmd = (dh, dr)
            
        t.add(s_old, action, reward, s)        
        ep_reward += reward
    
    
    return t, ep_reward

def rollout(episodes, env, model=None, sample_action=True, cmd=None, render=False, 
            replay_buffer=None, device=None, evaluation=False, epsilon=-1.0, max_return=300):
    """
    @param model: Model to user to select action. If None selects random action.
    @param cmd: If None will be sampled from the replay buffer.
    @param sample_action=True: If True samples action from distribution, otherwise 
                                selects max.
    @param epsilon - Probability of doing a random action. Between 0 and 1.0. Or -1 if no random actions are needed?
    """
    trajectories = []
    rewards = [] 
    length = 0
    
    for e in range(episodes):
        if (model is not None) and (cmd is None):
            if evaluation:
                cmd = replay_buffer.eval_command()
            else:
                cmd = replay_buffer.sample_command()
            
        t, reward = rollout_episode(env=env, model=model, sample_action=sample_action, cmd=cmd,
                            render=render, device=device, epsilon=epsilon)            
        
        trajectories.append(t)
        length += t.length
        rewards.append(reward)
    
    if render:
        env.close()
    
    return Rollout(episodes=episodes, trajectories=trajectories, rewards=rewards, length=length)

class Rollout(object):
    def __init__(self, episodes, trajectories, rewards, length):
        self.rewards = rewards
        self.length = length
        self.trajectories = trajectories
        self.episodes = episodes

    @property
    def mean_length(self):
        return self.length * 1.0 / self.episodes

    @property
    def mean_reward(self):
        return np.mean(self.rewards)


def to_training(s, dr, dh, return_scale=0.01, horizon_scale=0.01):
    l = s.tolist()
    l.append(dh*horizon_scale)
    l.append(dr*return_scale)
    return l

def save_checkpoint(path, model, optimizer, loss, updates, steps):
    
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'updates': updates,
            'steps': steps}, 
            path)
    
def load_checkpoint(path, model, optimizer, device, train=True):
    epoch = 0
    loss = 0.0    
    steps = 0
    updates = 0
    if os.path.exists(path):        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        updates = checkpoint['updates']
        steps = checkpoint['steps']
        print(f"Existing model found. Loading from epoch {epoch}, steps {steps} with loss: {loss}")
    else:
        print("No checkpoint found. Creating new model.")

    if train:
        model.train()
    else:
        model.eval()
    
    return Checkpoint(model, optimizer, loss, updates, steps)

class Checkpoint(object):
    def __init__(self, model, optimizer, loss, updates, steps):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.updates = updates
        self.steps = steps


class Trajectory(object):
    
    def __init__(self, horizon_scale=0.0001, return_scale=0.0001):
        self.trajectory = []
        self.cum_sum = []
        self.total_return = 0
        self.length = 0
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        
    def add(self, state, action, reward, state_prime):
        self.trajectory.append((state, action, reward, state_prime))
        if len(self.cum_sum) == 0:
            self.cum_sum.append(reward)
        else:
            self.cum_sum.append(self.cum_sum[len(self.cum_sum)-1] + reward)
        self.total_return += reward
        self.length += 1
    
    def sample_segment(self):
        T = self.length

        t1 = random.randint(1, T)
        t2 = T #random.randint(t1, T)
        
        state = self.trajectory[t1-1][0]
        action = self.trajectory[t1-1][1]

        d_r = self.cum_sum[t2 - 1] - self.cum_sum[t1 - 2]
        
        d_h = t2 - t1 + 1.0

        return ((state,d_r,d_h),action)
    
class ReplayBuffer(object):
    
    def __init__(self, max_size, last_few):
        """
        @param last_few: Number of episodes from the end of the replay buffer
        used for sampling exploratory commands.
        """
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = []
        
        self.last_few = last_few
        
    def add(self, trajectories):
        for trajectory in trajectories:
            self._add(trajectory)        
            
    def _add(self, trajectory):
        self.buffer.append(trajectory)
        
        self.buffer = sorted(self.buffer, key=lambda x: x.total_return, reverse=True)
        self.buffer = self.buffer[:self.max_size]
    
    def sample(self, batch_size, device):
        trajectories = np.random.choice(self.buffer, batch_size, replace=True)
        x = []
        y = []
        for t in trajectories:
            
            segment = t.sample_segment()
            
                        
            
            (s, dr, dh), action = segment
            l = to_training(s, dr, dh)
            
            x.append(l)
            y.append(action)
            
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        return x, y
    
    def sample_command(self):
        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        m = np.mean([e.total_return for e in eps])
        s = np.std([e.total_return for e in eps])
#         s = median_absolute_deviation([e.total_return for e in eps])
        
        dr_0 = np.random.uniform(m, m + s)
        
        return dh_0, dr_0

    def eval_command(self):
        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        dr_0 = np.min([e.total_return for e in eps])
        
        return dh_0, dr_0
