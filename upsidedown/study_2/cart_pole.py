import gym
import os
import numpy as np
import random
import torch.nn as nn
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from experiment import rollout, ReplayBuffer, Trajectory, load_checkpoint, save_checkpoint
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment()

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

@ex.command
def train(_run, experiment_name, checkpoint_path, batch_size, hidden_size, solved_mean_reward, solved_n_episodes, max_steps, 
    replay_size, last_few, n_warmup_episodes, n_episodes_per_iter, n_updates_per_iter,
    epsilon, eval_episodes, max_return):
    """
    Begin or resume training a policy.
    """
    log_dir = f'runs/{_run._id}_{experiment_name}'
    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make('CartPole-v1')

    loss_object = torch.nn.BCEWithLogitsLoss().to(device)

    d = env.observation_space.shape[0]
        
    model = Behavior(hidden_size=hidden_size, input_shape=d+2, num_actions=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)

    # Random rollout
    roll = rollout(episodes=n_warmup_episodes, env=env, render=False, max_return=max_return)
    rb.add(roll.trajectories)
    print(f"Mean Episode Reward: {roll.mean_reward}")

    # Keep track of steps used during random rollout!
    c = load_checkpoint(checkpoint_path, model, optimizer, device, train=True)
    updates, steps, loss  = c.updates, c.steps, c.loss

    steps += roll.length

    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, loss=loss, updates=updates, steps=steps)

    # Plot initial values
    writer.add_scalar('Train/reward', roll.mean_reward, steps)   
    writer.add_scalar('Train/length', roll.mean_length, steps)

    loss_sum = 0
    loss_count = 0    
    rewards = [] # For stopping   

    while True:
        for _ in range(n_updates_per_iter):
            updates +=1 

            x, y = rb.sample(batch_size, device)    
            loss = train_step(x, y, model, optimizer, loss_object)
            loss_sum += loss
            loss_count += 1
            writer.add_scalar('Loss/loss', loss, updates)
            

        # Save updated model
        avg_loss = loss_sum/loss_count
        print(f'u: {updates}, s: {steps}, Loss: {avg_loss}')

        save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, loss=avg_loss, updates=updates, steps=steps)

        # Exploration
        roll = rollout(n_episodes_per_iter, env=env, 
            model=model, sample_action=True, replay_buffer=rb, 
            device=device, action_fn=action_fn, epsilon=epsilon, max_return=max_return)
        rb.add(roll.trajectories)
        
        steps += roll.length

        (dh, dr) = rb.sample_command()
        writer.add_scalar('Train/dr', dr, steps)
        writer.add_scalar('Train/dh', dh, steps)
        
        writer.add_scalar('Train/reward', roll.mean_reward, steps)
        writer.add_scalar('Train/length', roll.mean_length, steps)

        # Eval
        roll = rollout(eval_episodes, env=env, model=model, 
                sample_action=True, replay_buffer=rb, 
                device=device, action_fn=action_fn, evaluation=True, 
                max_return=max_return)

        (dh, dr) = rb.eval_command()
        writer.add_scalar('Eval/dr', dr, steps)
        writer.add_scalar('Eval/dh', dh, steps)

        print(f"Eval Episode Mean Reward: {roll.mean_reward}")        
        writer.add_scalar('Eval/reward', roll.mean_reward, steps)        
        writer.add_scalar('Eval/length', roll.mean_length, steps)

        # Stopping criteria
        rewards.extend(roll.rewards)
        rewards = rewards[-solved_n_episodes:]
        eval_mean_reward = np.mean(rewards)

        if eval_mean_reward >= solved_mean_reward:
            print(f"Task considered solved. Achieved {eval_mean_reward} >= {solved_mean_reward} over {solved_n_episodes} episodes.")
            break
        
        if steps >= max_steps:
            print(f"Steps {steps} exceeds max env steps {max_steps}. Stopping.")
            break        
        
    add_artifact()

@ex.capture
def add_artifact(checkpoint_path):
    ex.add_artifact(checkpoint_path, name='checkpoint.pt')

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

@ex.command
def play(checkpoint_path, epsilon, sample_action, hidden_size, play_episodes, max_return, dh, dr):
    """
    Play episodes using a trained policy. 
    """
    env = gym.make('CartPole-v1')
    d = env.observation_space.shape[0]
    model = Behavior(hidden_size=hidden_size, input_shape=d+2, num_actions=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # rb.sample_command()
    # dh       dr

    cmd = (dh, dr)
    # cmd = rb.sample_command()
   
    checkpoint = load_checkpoint(checkpoint_path, train=False, 
        model=model, optimizer=optimizer, device=device)
    updates = checkpoint.updates
    steps   = checkpoint.steps
    loss    = checkpoint.loss

    print(model)
    print(f"Loaded model at update: {updates}, steps: {steps} with loss {loss}")

    for _ in range(play_episodes):
        roll = rollout(episodes=1, env=env, model=model, 
            sample_action=sample_action, cmd=cmd, render=True, device=device, 
            action_fn=action_fn, epsilon=epsilon, max_return=max_return)

        print(f"Episode Reward: {roll.mean_reward}, Length: {roll.length}")
    
@ex.config
def run_config():
    hidden_size = 32
    max_return = 300 # Max return per episode 
    epsilon = 0.0
    
    # Train specific
    batch_size = 1024
    solved_mean_reward = 195 # Considered solved when the average reward is greater than or equal to
    solved_n_episodes  = 100 #  195.0 over 100 consecutive trials.
    max_steps = 10**7
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_warmup_episodes = 50
    n_episodes_per_iter = 10
    n_updates_per_iter = 100
    eval_episodes = 10

    experiment_name = f'cartpole_hs{hidden_size}_mr{max_return}_b{batch_size}_rs{replay_size}_lf{last_few}_nw{n_warmup_episodes}_ne{n_episodes_per_iter}_nu{n_updates_per_iter}_e{epsilon}_ev{eval_episodes}'
    checkpoint_path = f'checkpoint_{experiment_name}.pt'

    # Play specific
    sample_action = True
    play_episodes = 5
    dh = 200
    dr = 300


@ex.automain
def main():
    """
    Default runs train() command.
    """
    train()