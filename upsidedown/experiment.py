import numpy as np
import torch

import os
import torch.nn.functional as F
import time 

import torch.nn as nn
           

def rollout_episode(env, model, sample_action=True, cmd=None, 
                    render=False, device=None, action_fn=None):
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
            action = action_fn(model, inputs, sample_action)
            
        if render:
            env.render()
            time.sleep(0.01)
            
        s_old = s        
        s, reward, done, info = env.step(action)
        if model is not None:
            dh = dh - 1
            dr = dr - reward
            cmd = (dh, dr)
            
        t.add(s_old, action, reward, s)        
        ep_reward += reward
    
    
    return t, ep_reward

def rollout(episodes, env, model=None, sample_action=True, cmd=None, render=False, 
            replay_buffer=None, device=None, action_fn=None):
    """
    @param model: Model to user to select action. If None selects random action.
    @param cmd: If None will be sampled from the replay buffer.
    @param sample_action=True: If True samples action from distribution, otherwise 
                                selects max.
    """
    trajectories = []
    rewards = [] 
    
    for e in range(episodes):
        if (model is not None) and (cmd is None):
            cmd = replay_buffer.sample_command()
            
        t, reward = rollout_episode(env=env, model=model, sample_action=sample_action, cmd=cmd,
                            render=render, device=device, action_fn=action_fn)            
        
        trajectories.append(t)
    
        rewards.append(reward)
    
    if render:
        env.close()
    
    return trajectories, np.mean(rewards)

def to_training(s, dr, dh):
    l = s.tolist()
    l.append(dr)
    l.append(dh)
    return l

class Behavior(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Behavior, self).__init__()
        self.fc1 = nn.Linear(input_shape,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,512)
        self.fc5 = nn.Linear(512,num_actions)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        output = self.fc5(output)
        return output
    


def save_model(name, epoch, model, optimizer, loss):
    path = f'{name}.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 
            path)
    
def load_model(name, model, optimizer, device, train=True):
    epoch = 0
    loss = 0.0
    path = f'{name}.pt'
    if os.path.exists(path):        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Existing model found. Loading from epoch {epoch} with loss: {loss}")
    else:
        print("No checkpoint found. Creating new model.")

    if train:
        model.train()
    else:
        model.eval()
    
    return epoch, model, optimizer, loss 


class Trajectory(object):
    
    def __init__(self):
        self.trajectory = []
        self.total_return = 0
        self.length = 0
        
    def add(self, state, action, reward, state_prime):
        self.trajectory.append((state, action, reward, state_prime))
        self.total_return += reward
        self.length += 1
    
    def sample_segment(self):
        T = len(self.trajectory)

        t1 = np.random.randint(1, T+1)
        t2 = np.random.randint(t1, T+1)

        state = self.trajectory[t1-1][0]
        action = self.trajectory[t1-1][1]

        d_r = 0.0
        for i in range(t1, t2 + 1):
            d_r += self.trajectory[i-1][2]

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
        
        dr_0 = np.random.uniform(m, m+s)
        
        return dh_0, dr_0
