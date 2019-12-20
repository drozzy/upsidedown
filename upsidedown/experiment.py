import numpy as np
import torch
PATH = 'cartpole.pt'
import os
import torch.nn.functional as F
import time 

import torch.nn as nn
def rollout_episode(env, model, sample_action=True, cmd=None, 
                    render=False, device=None):
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

            action_probs = model(inputs)
            action_probs = torch.sigmoid(action_probs) #, dim=-1)

            if sample_action:                
                m = torch.distributions.bernoulli.Bernoulli(probs=action_probs)            
                action = int(m.sample().squeeze().cpu().numpy())
            else:
                action = int(np.round(action_probs.detach().squeeze().numpy()))

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
            replay_buffer=None, device=None):
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
                            render=render, device=device)            
        
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

def segments_to_training(segments, device):
    x = []
    y = []
    for (s, dr, dh), action in segments:
        l = to_training(s, dr, dh)
        x.append(l)
        y.append(action)
        
    x = torch.tensor(x).float().to(device)
    y = torch.tensor(y).float().to(device)
    
    return x, y

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
    


def save_model(epoch, model, optimizer, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 
            PATH)
    
def load_model(env, device, train=True):
    d = env.observation_space.shape[0]
    model = Behavior(input_shape=d+2, num_actions=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    epoch = 0
    loss = 0.0
    
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

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
            
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)

        return x, y
    
    def sample_command(self):
        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        m = np.mean([e.total_return for e in eps])
        s = np.std([e.total_return for e in eps])
        
        dr_0 = np.random.uniform(m, m+s)
        
        return dh_0, dr_0
