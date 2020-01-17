from collections import namedtuple
import torch.nn.functional as F
import time 
import gym
from gym.wrappers.time_limit import TimeLimit
from itertools import cycle
import os
from gym.wrappers.frame_stack import FrameStack
import numpy as np
import random
import datetime
import torch
from torch import nn
from itertools import count

# Command expressing desired quantities (numpy, single elements) 
Command = namedtuple('Command', ['dr', 'dh'])

# Batch-sized sample from the replay buffer, each element already a torch.tensor on device
Sample = namedtuple('Sample', ['prev_action', 'state', 'dr', 'dh', 'action'])

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
        
    def state_dict(self):
        return {'max_size' : self.max_size, 'cur_size' : self.cur_size, 'buffer': self.buffer, 'last_few': self.last_few}

    def load_state_dict(state_dict):
        self.max_size = state_dict['max_size']
        self.cur_size = state_dict['cur_size']
        self.buffer = state_dict['buffer']
        self.last_few = state_dict['last_few']

    def add(self, trajectories):
        for trajectory in trajectories:
            self._add(trajectory)       

    def clear(self):
        self.cur_size = 0
        self.buffer = [] 
            
    def _add(self, trajectory):
        self.buffer.append(trajectory)
        
        self.buffer = sorted(self.buffer, key=lambda x: x.total_return, reverse=True)
        self.buffer = self.buffer[:self.max_size]

    @property
    def stats(self):
        episodes = self.buffer[:self.last_few]

        mean_last = np.mean([e.total_return for e in episodes])
        std_last =  np.std([e.total_return for e in episodes])

        mean = np.mean([e.total_return for e in self.buffer])
        std =  np.std([e.total_return for e in self.buffer])        

        mean_len = np.mean([e.length for e in self.buffer])
        std_len = np.std([e.length for e in self.buffer])
        mean_len_last = np.mean([e.length for e in episodes])
        std_len_last = np.std([e.length for e in episodes])

        return mean, std, mean_last, std_last, mean_len, std_len, mean_len_last, std_len_last

    def sample(self, batch_size, device):
        trajectories = np.random.choice(self.buffer, batch_size, replace=True)
        prev_action_batch = []
        action_batch = []
        s_batch  = []
        dr_batch = []
        dh_batch = []

        for t in trajectories:
            
            segment = t.sample_segment()

            s_batch.append(segment.state)
            dr_batch.append(segment.dr)
            dh_batch.append(segment.dh)
            prev_action_batch.append(segment.prev_action)
            
            action_batch.append(segment.action)
            if len(s_batch) >= batch_size:
                break

        s_batch = torch.tensor(s_batch).float().to(device)
        dr_batch = torch.tensor(dr_batch).unsqueeze(dim=1).to(device)
        dh_batch = torch.tensor(dh_batch).unsqueeze(dim=1).to(device)
        prev_action_batch = torch.tensor(prev_action_batch).long().to(device)

        action_batch = torch.tensor(action_batch).to(device)

        return Sample(prev_action=prev_action_batch, state=s_batch, dr=dr_batch, dh=dh_batch, action=action_batch)
    
    def sample_command(self, init_dr, init_dh):
        # Special case: when there is no experience yet
        if len(self.buffer) == 0:
            return Command(dr=init_dr, dh=init_dh)

        episodes = self.buffer[:self.last_few]
        
        # This seems to work for cartpole:
        # dh_0 = 2 * np.max([e.length for e in episodes])
        # max_return = np.max([e.total_return for e in episodes])
        # dr_0 = max(2 * max_return, 1)

        # This seems to work for lunar-lander
        dh_0 = np.mean([e.length for e in episodes])
        m = np.mean([e.total_return for e in episodes])
        s = np.std([e.total_return for e in episodes])        
        dr_0 = np.random.uniform(m, m + s)

        return Command(dh=dh_0, dr=dr_0)

    def eval_command(self):
        episodes = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in episodes])
        
        dr_0 = np.min([e.total_return for e in episodes])
        
        return Command(dh=dh_0, dr=dr_0)
