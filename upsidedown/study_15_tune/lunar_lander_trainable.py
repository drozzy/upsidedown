from ray.tune import Trainable
from lib import ReplayBuffer, Command, Trajectory, Rollout
from model import Behavior
from collections import namedtuple
import torch.nn.functional as F
import time 
import math
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

class LunarLanderTrainable(Trainable):
    def _setup(self, config={}):
        self.config = self.default_config
        self.config.update(config)

        self.seed =  self.config['seed']
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Fill in from config
        self.env_name = self.config['env_name']
        self.num_stack = self.config['num_stack']
        self.hidden_size = self.config['hidden_size']
        self.lr = self.config['lr']
        self.lr_end = self.config['lr_end']
        self.lr_decay = self.config['lr_decay']
        self.replay_size = self.config['replay_size']
        self.last_few = self.config['last_few']
        self.n_episodes_per_iter = self.config['n_episodes_per_iter']
        self.n_updates_per_iter = self.config['n_updates_per_iter']
        self.epsilon = self.config['epsilon']
        self.max_return = self.config['max_return']        
        self.render = self.config['render']
        self.return_scale = self.config['return_scale']
        self.horizon_scale = self.config['horizon_scale']
        self.init_dr = self.config['init_dr']
        self.init_dh = self.config['init_dh']
        self.batch_size = self.config['batch_size']
        self.eval_episodes = self.config['eval_episodes']
        self.max_steps = self.config['max_steps']
        self.solved_min_reward = self.config['solved_min_reward']
        self.solved_n_episodes = self.config['solved_n_episodes']
        self.epsilon_decay = self.config['epsilon_decay']

        # Initialize 
        self.device = torch.device("cpu")
        self.steps = 0
        self.loss = None
        self.rewards = []
        self.last_eval_step = 0

        self.env =  FrameStack(gym.make(self.env_name), num_stack=self.num_stack)
        self.env.seed(self.seed)
        self.loss_object = torch.nn.CrossEntropyLoss().to(self.device)

        self.model = Behavior(hidden_size=self.hidden_size, state_shape=self.env.observation_space.shape, num_actions=self.env.action_space.n,
            return_scale=self.return_scale, horizon_scale=self.horizon_scale).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.annealed_lr)

        self.rb = ReplayBuffer(max_size=self.replay_size, last_few=self.last_few)
    

    def play(self, dr=300, dh=300, sample_action=True, play_episodes=5):
        """
        Play a few episodes with the currently trained policy.
        """
        cmd = Command(dr=dr, dh=dh)

        for _ in range(play_episodes):
            roll = self.rollout(episodes=1, sample_action=sample_action, 
                                  cmd=cmd, render=True)

            print(f"Episode Reward: {roll.mean_reward} Steps: {roll.length}")

    def _train(self):
        """
        Do one iteration of training
        """
        return self.do_iteration()

    def do_iteration(self):
        results = {}

        #### Exloration ####
        print("Begining Exploration.")
        dr, dh, steps, mean_reward, mean_length = self.do_exploration()
        self.steps += steps
        
        results['Exploration/dr'] = dr
        results['Exploration/dh'] = dh
        results['Exploration/reward_mean'] = mean_reward
        results['Exploration/length_mean'] = mean_length
        results['Exploration/steps'] = steps
        # Special value for ray/tune
        results['timesteps_this_iter'] = steps
        
        #### Updates ####
        print("Begining Updates.")
        mean_loss = self.do_updates()
            
        results['Updates/mean_loss'] = mean_loss
        # Special value for ray/tune
        results['mean_loss'] = mean_loss

        #### Evaluation ####
        print("Begining Eval.")
        done_training, eval_dr, eval_dh, eval_mean_reward, eval_mean_length = self.do_eval()

        results['Eval/dr'] = eval_dr
        results['Eval/dh'] = eval_dh
        results['Eval/reward_mean'] = eval_mean_reward        
        results['Eval/length_mean'] = eval_mean_length
        # Special value for ray/tune
        results['episode_reward_mean'] = eval_mean_reward
        results['done'] = done_training

        mean, std, mean_last, std_last, mean_len, std_len, mean_len_last, std_len_last = self.rb.stats()
        
        results['Buffer/reward_mean'] = mean
        results['Buffer/reward_std'] = std
        results['Buffer/last_few_reward_mean'] = mean_last
        results['Buffer/last_few_reward_std'] = std_last

        results['Buffer/length_mean'] = mean_len
        results['Buffer/length_std'] = std_len
        results['Buffer/last_few_length_mean'] = mean_len_last
        results['Buffer/last_few_length_std'] = std_len_last

        results['Params/lr'] = self.annealed_lr
        results['Params/last_few'] = self.last_few
        results['Params/epsilon'] = self.annealed_epsilon

        return results

    @property
    def annealed_epsilon(self):
        # Anneal epsilon
        
        EPS_START = self.epsilon
        EPS_END = 0.0
        EPS_DECAY = self.epsilon_decay

        sample = random.random()
        annealed = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps / EPS_DECAY)
        
        return annealed

    @property
    def annealed_lr(self):
        # Anneal epsilon
        
        EPS_START = self.lr
        EPS_END =   self.lr_end
        EPS_DECAY = self.lr_decay

        sample = random.random()
        annealed = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps / EPS_DECAY)
        
        return annealed


    def do_exploration(self):
        # Sample command for the exploration
        exploration_cmd = self.rb.sample_command(self.init_dr, self.init_dh)

        # NOW CLEAR the buffer
        # self.rb.clear()

        # Exploration    
        roll = self.rollout(episodes=self.n_episodes_per_iter, cmd=exploration_cmd, sample_action=True, epsilon=self.annealed_epsilon)
        self.rb.add(roll.trajectories)

        steps = roll.length
        return exploration_cmd.dr, exploration_cmd.dh, roll.length, roll.mean_reward, roll.mean_length

    def do_updates(self):
        loss_sum = 0
        loss_count = 0
       
        for _ in range(self.n_updates_per_iter):
            sample = self.rb.sample(self.batch_size, self.device)    

            loss = self.train_step(sample)
            loss_sum += loss
            loss_count += 1            

        avg_loss = loss_sum/loss_count

        return avg_loss

    def do_eval(self):
        
        eval_cmd = self.rb.eval_command()

        roll = self.rollout(episodes=self.eval_episodes, epsilon=0.0,
                sample_action=True, cmd=eval_cmd)

        steps_exceeded = self.steps >= self.max_steps

        self.last_eval_step = self.steps

        cmd = self.rb.eval_command()
        
        # Stopping criteria
        self.rewards.extend(roll.rewards)
        self.rewards = self.rewards[-self.solved_n_episodes:]
        eval_min_reward = np.min(self.rewards)

        solved = eval_min_reward >= self.solved_min_reward
        if solved:
            print(f"Task considered solved. Achieved {eval_min_reward} >= {self.solved_min_reward} over {self.solved_n_episodes} episodes.")
        
        done_training = solved or steps_exceeded

        return done_training, cmd.dr, cmd.dh, roll.mean_reward, roll.mean_length

    def train_step(self, sample):
        self.optimizer.zero_grad()    
        predictions = self.model(prev_action=sample.prev_action, state=sample.state, dr=sample.dr, dh=sample.dh)
        loss = self.loss_object(predictions, sample.action)

        loss.backward()
        self.optimizer.step()
        
        return loss.detach().numpy()

    def rollout(self, episodes, epsilon, sample_action=True, cmd=None, render=False):
        assert cmd is not None

        trajectories = []
        rewards = [] 
        length = 0

        for e in range(episodes):
            t, reward = self.rollout_episode(sample_action=sample_action, cmd=cmd, render=render, epsilon=epsilon)
            
            trajectories.append(t)
            length += t.length
            rewards.append(reward)
        
        if render:
            self.env.close()
        
        return Rollout(episodes=episodes, trajectories=trajectories, rewards=rewards, length=length)

    def rollout_episode(self, sample_action, cmd, render, epsilon):
        """
        @param sample_action=True: If True samples action from distribution, otherwise 
                                    selects max.
        @param epsilon - Probability of doing a random action. Between 0 and 1.0.
        """
        s = self.env.reset()
        done = False
        ep_reward = 0.0
        prev_action = self.env.action_space.n # First action is "special" start of episode action

        t = Trajectory()
        
        while not done:
            with torch.no_grad():
                self.model.eval()
                action = self.get_action(prev_action=prev_action, state=s, cmd=cmd, sample_action=sample_action, epsilon=epsilon)
                self.model.train()
            
            if render:
                env.render()
                time.sleep(0.01)
                
            s_old = s        
            s, reward, done, info = self.env.step(action)
            
            dh = max(cmd.dh - 1, 1)
            dr = cmd.dr - reward
            cmd = Command(dr=dr, dh=dh)
                
            t.add(prev_action, s_old, action, reward, s)    
            prev_action = action    
            ep_reward += reward

        return t, ep_reward

    def get_action(self, prev_action, state, cmd, sample_action, epsilon):
        if random.random() < epsilon: # Random action
            return self.env.action_space.sample()

        prev_action = torch.tensor(prev_action).long().unsqueeze(dim=0).to(self.device)
        dr = torch.tensor([cmd.dr]).float().unsqueeze(dim=0).to(self.device)
        dh = torch.tensor([cmd.dh]).float().unsqueeze(dim=0).to(self.device)
        state = torch.tensor(state).float().unsqueeze(dim=0).to(self.device)
        action_logits = self.model(prev_action=prev_action, state=state, dr=dr, dh=dh)
        action_probs = torch.softmax(action_logits, axis=-1)

        if sample_action:        
            m = torch.distributions.categorical.Categorical(logits=action_logits)             
            action = int(m.sample().squeeze().cpu().numpy())        
        else:
            action = int(np.argmax(action_probs.detach().squeeze().numpy()))
        return action

    def _restore(self, checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.rb.load_state_dict(checkpoint['replay_buffer'])
            self.steps = checkpoint['steps']
            self.rewards = checkpoint['rewards']

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'replay_buffer' : self.rb.state_dict(),
                'rewards': self.rewards,
                'steps': self.steps}, 
                checkpoint_path)
        return checkpoint_path

    @property
    def default_config(self):
        # Environment to train on
        return {
            'seed' : None,
            'env_name': 'LunarLander-v2',
            'num_stack' : 4,
            'hidden_size' : 32,

            # Starting epsilon value for exploration
            'epsilon' : 0.99,
            # how fast to decay the epsilon to zero (X-value decays epsilon in approximately 10X steps. E.g. 100_000 decay reduces it to zero in 1_000_000 steps)            
            'epsilon_decay' : 100_000,

            'return_scale': 0.01,
            'horizon_scale' : 0.001,
            'lr': 0.1,
            'lr_end': 0.0001,
            'lr_decay' : 500_000,
            'batch_size' : 1024,

            # Solved when min reward is at least this ...
            'solved_min_reward' : 200,
            # ... over this many episodes
            'solved_n_episodes' :  100,
            'max_steps' : 10**7,

            # Maximum size of the replay buffer in episodes
            'replay_size' : 512,
            'n_episodes_per_iter' : 8,

            # How many last episodes to use for selecting the desire/horizon from
            'last_few' : 16,

            # How many updates of the model to do by sampling from the replay buffer
            'n_updates_per_iter' : 128,
            'eval_episodes' : 10,

            # The max return value for any given episode (TODO: make sure it's used correctly)
            'max_return' : 300,

            # Initial dh, dr values to use when our buffer is empty
            'init_dh' : 1,
            'init_dr' : 0,
            'render' : False
        }