import gym
import os
import numpy as np
import random
import torch.nn as nn
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from experiment import rollout, ReplayBuffer, Trajectory, load_model, save_model
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment()

MODEL_NAME = 'model_cart_pole_v1'

class Behavior(torch.nn.Module):
    def __init__(self, hidden_size, input_shape, num_actions):
        super(Behavior, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Linear(input_shape, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )        

    def forward(self, x):
        return self.classifier(x)

@ex.config
def train_config():
    batch_size = 1024
    solved_mean_reward = 350 # officially 195 - be we sample less episode
    max_steps = 10**7
    hidden_size = 32
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_warmup_episodes = 50
    n_episodes_per_iter = 10
    n_updates_per_iter = 100
    start_epsilon = 0.1

@ex.capture
def run_train(batch_size, hidden_size, solved_mean_reward, max_steps, 
    replay_size, last_few, n_warmup_episodes, n_episodes_per_iter, n_updates_per_iter,
    start_epsilon):
    writer = SummaryWriter()
    env = gym.make('CartPole-v1')

    loss_object = torch.nn.BCEWithLogitsLoss().to(device)

    d = env.observation_space.shape[0]
        
    model = Behavior(hidden_size=hidden_size, input_shape=d+2, num_actions=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)

    # Random rollout
    trajectories, mean_reward, length = rollout(episodes=n_warmup_episodes, env=env, render=False)
    rb.add(trajectories)

    print(f"Mean Episode Reward: {mean_reward}")

    # Keep track of steps used during random rollout!
    epoch, model, optimizer, loss, steps = load_model(MODEL_NAME, model, optimizer, device, train=True)
    steps += length
    save_model(MODEL_NAME, epoch, model, optimizer, loss, steps)

    # Plot initial values
    writer.add_scalar('Steps/reward', mean_reward, steps)       

    loss_sum = 0
    loss_count = 0       

    epoch, model, optimizer, loss, steps = load_model(MODEL_NAME, model, optimizer, device, train=True)
    
    for i in count(start=epoch):
        x, y = rb.sample(batch_size, device)    
        loss = train_step(x, y, model, optimizer, loss_object)
        loss_sum += loss
        loss_count += 1
        writer.add_scalar('Loss/loss', loss, i)
        
        (dh, dr) = rb.sample_command()
        writer.add_scalar('Epoch/dh', dh, i)
        writer.add_scalar('Epoch/dr', dr, i)
       
        if i % n_updates_per_iter == 0:        
            trajectories, mean_reward, length = rollout(n_episodes_per_iter, env=env, 
                model=model, sample_action=True, replay_buffer=rb, 
                device=device, action_fn=action_fn, epsilon=start_epsilon)
            rb.add(trajectories)
            
            print(f"Average Episode Reward: {mean_reward}")  
            
            steps += length
            avg_loss = loss_sum/loss_count
            save_model(MODEL_NAME, i, model, optimizer, avg_loss, steps)        
            print(f"Average Episode Reward: {mean_reward}")        
            writer.add_scalar('Steps/reward', mean_reward, steps)
            
            mean_length = length*1.0/n_episodes_per_iter
            writer.add_scalar('Steps/length', mean_length, steps)
            
            if mean_reward >= solved_mean_reward:
                print("Task considered solved! Stopping.")
                break
            
            if steps >= max_steps:
                print(f"Steps {steps} exceeds max env steps {max_steps}. Stopping.")
                break

        if i % 200 == 0:
            avg_loss = loss_sum/loss_count
            print(f'i: {i}, s: {steps}, Loss: {avg_loss}')
            save_model(MODEL_NAME, i, model, optimizer, avg_loss, steps)    

def train_step(inputs, targets, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model(inputs)
    targets = targets.float()
    loss = loss_object(predictions, targets.unsqueeze(1))
    
    loss.backward()
    optimizer.step()
    
    return loss

def action_fn(env, model, inputs, sample_action, epsilon):
    action_probs = model(inputs)
    action_probs = torch.sigmoid(action_probs)
    
    if random.random() < epsilon:
        return env.action_space.sample()

    if sample_action:
        m = torch.distributions.bernoulli.Bernoulli(probs=action_probs)            
        action = int(m.sample().squeeze().cpu().numpy())
    else:
        action = int(np.round(action_probs.detach().squeeze().numpy()))
    return action

@ex.capture
def run_play(epsilon, sample_action):
    env = gym.make('CartPole-v1')
    d = env.observation_space.shape[0]
    model = Behavior(input_shape=d+2, num_actions=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # rb.sample_command()
    # dh       dr

    dh = 100
    dr = 400
    cmd = (dh, dr)
    # cmd = rb.sample_command()
   
    e, model, _, l, steps = load_model(MODEL_NAME, train=False, 
        model=model, optimizer=optimizer, device=device)
    print(f"Loaded model at epoch: {e} with loss {l}")
    _, mean_reward, length = rollout(episodes=3, env=env, model=model, 
        sample_action=sample_action, cmd=cmd, render=True, device=device, 
        action_fn=action_fn, epsilon=epsilon)

    print(f"Average Episode Reward: {mean_reward}, Mean Length: {length}")

@ex.config
def play_config():
    epsilon = 0.0
    sample_action = True

@ex.config
def run_config():
    train = True

@ex.automain
def main(train):
    if train:
        run_train()
    else:
        run_play()