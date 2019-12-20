import numpy as np
import torch

import os
import torch.nn.functional as F
import time 
import random
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
        self.cum_sum = []
        self.total_return = 0
        self.length = 0
        
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
        t2 = random.randint(t1, T)

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
        import time
        sample_t = 0.0
        after_t = 0.0
        for t in trajectories:
            start = time.time()
            segment = t.sample_segment()
            sample_t += time.time() - start
                        
            start = time.time()
            (s, dr, dh), action = segment
            l = to_training(s, dr, dh)
            after_t += time.time() - start
            x.append(l)
            y.append(action)
            
            
#         print(f'Sample time: {sample_t}')
#         print(f'After time: {after_t}')
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        return x, y, sample_t
    
    def sample_command(self):
        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        m = np.mean([e.total_return for e in eps])
        s = np.std([e.total_return for e in eps])
        
        dr_0 = np.random.uniform(m, m+s)
        
        return dh_0, dr_0
