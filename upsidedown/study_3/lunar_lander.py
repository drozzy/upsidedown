import gym
import os
import numpy as np
import random
import torch
from torch import nn
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from experiment import rollout, ReplayBuffer, Trajectory, load_checkpoint, save_checkpoint
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sacred.stflow import LogFileWriter

ex = Experiment()

class Behavior(nn.Module):
    def __init__(self, hidden_size, state_shape, cmd_shape, num_actions):
        super(Behavior, self).__init__()
        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_cmd = nn.Linear(cmd_shape, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        output_spate = self.fc_state(x[0])
        output_cmd = torch.sigmoid(self.fc_cmd(x[1]))
        
        output = output_spate * output_cmd
        
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output

@ex.command
def train(_run, experiment_name, hidden_size, replay_size, last_few, lr):
    """
    Begin or resume training a policy.
    """
    log_dir = f'tensorboard/{_run._id}_{experiment_name}'
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make('LunarLander-v2')
    
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    
    model = Behavior(hidden_size=hidden_size, state_shape=env.observation_space.shape[0], cmd_shape=2, num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)

    do_train(env=env, model=model, optimizer=optimizer, loss_object=loss_object, rb=rb, writer=writer)
   
@ex.capture 
def do_train(env, model, optimizer, loss_object, rb, writer, checkpoint_path, 
    n_warmup_episodes, max_return):

    # Random rollout
    roll = rollout(episodes=n_warmup_episodes, env=env, render=False, max_return=max_return)
    rb.add(roll.trajectories)
    print(f"Mean Episode Reward: {roll.mean_reward}")

    # Keep track of steps used during random rollout!
    print("Trying to load:")
    print(checkpoint_path)
    c = load_checkpoint(checkpoint_path, model, optimizer, device, train=True)
    updates, steps, loss = c.updates, c.steps, c.loss

    steps += roll.length
    
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, loss=loss, updates=updates, steps=steps)

    # Plot initial values
    writer.add_scalar('Train/reward', roll.mean_reward, steps)   
    writer.add_scalar('Train/length', roll.mean_length, steps)

    do_iterations(env, model, optimizer, loss_object, rb, writer)

@ex.capture
def do_iterations(env, model, optimizer, loss_object, rb, writer, checkpoint_path):
    
    print("Trying to load:")
    print(checkpoint_path)
    c = load_checkpoint(checkpoint_path, model, optimizer, device, train=True)
    updates, steps, loss = c.updates, c.steps, c.loss

    done = False
    while not done:
        updates, steps, done = do_iteration(env, model, optimizer, loss_object, rb, writer, updates=updates, 
            steps=steps)

    add_artifact()

@ex.capture
def do_exploration(env, model, rb, writer, steps, n_episodes_per_iter, epsilon, max_return):

    # Exploration    
    roll = rollout(n_episodes_per_iter, env=env, model=model, 
        sample_action=True, replay_buffer=rb, device=device, 
        epsilon=epsilon, max_return=max_return)
    rb.add(roll.trajectories)

    steps += roll.length
    
    (dh, dr) = rb.sample_command()
    writer.add_scalar('Exploration/dr', dr, steps)
    writer.add_scalar('Exploration/dh', dh, steps)

    writer.add_scalar('Exploration/reward', roll.mean_reward, steps)
    writer.add_scalar('Exploration/length', roll.mean_length, steps)

    return steps

@ex.capture
def do_iteration(env, model, optimizer, loss_object, rb, writer, updates, steps, checkpoint_path, batch_size, max_steps, 
    solved_min_reward, solved_n_episodes, n_episodes_per_iter, n_updates_per_iter, 
    epsilon, eval_episodes, eval_every_n_steps, max_return):

    steps = do_exploration(env, model, rb, writer, steps)
    
    # Updates    
    loss_sum = 0
    loss_count = 0
    rewards = []
    last_eval_step = 0

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
    
    # Eval
    steps_exceeded = steps >= max_steps
    time_to_eval = ((steps - last_eval_step) >= eval_every_n_steps) or steps_exceeded

    if time_to_eval:
        last_eval_step = steps

        roll = rollout(eval_episodes, env=env, model=model, 
                sample_action=True, replay_buffer=rb, 
                device=device, evaluation=True,
                max_return=max_return)

        (dh, dr) = rb.eval_command()
        writer.add_scalar('Eval/dr', dr, steps)
        writer.add_scalar('Eval/dh', dh, steps)
        
        writer.add_scalar('Eval/reward', roll.mean_reward, steps) 
        writer.add_scalar('Eval/length', roll.mean_length, steps)
        
        print(f"Eval Episode Mean Reward: {roll.mean_reward}")      

        # Stopping criteria
        rewards.extend(roll.rewards)
        rewards = rewards[-solved_n_episodes:]
        eval_min_reward = np.min(rewards)

        if eval_min_reward >= solved_min_reward:
            print(f"Task considered solved. Achieved {eval_min_reward} >= {solved_min_reward} over {solved_n_episodes} episodes.")
            return updates, steps, True
    
    if steps_exceeded:
        print(f"Steps {steps} exceeds max env steps {max_steps}. Stopping.")
        return updates, steps, True

    return updates, steps, False  


@ex.capture
def add_artifact(checkpoint_path):
    ex.add_artifact(checkpoint_path, name='checkpoint.pt')
           
def train_step(inputs, targets, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model([inputs[:, :-2], inputs[:, -2:]])
    loss = loss_object(predictions, targets)
    
    loss.backward()
    optimizer.step()
    
    return loss



@ex.command
def play(checkpoint_path, epsilon, sample_action, hidden_size, play_episodes, dh, dr):
    """
    Play episodes using a trained policy. 
    """
    env = gym.make('LunarLander-v2')
    cmd = (dh, dr)

    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape[0], cmd_shape=2, num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    c = load_checkpoint(name=checkpoint_path, train=False, 
        model=model, optimizer=optimizer, device=device)

    for _ in range(play_episodes):
        roll = rollout(episodes=1, env=env, model=model, sample_action=sample_action, 
                              cmd=cmd, render=True, device=device)

        print(f"Episode Reward: {roll.mean_reward}")


@ex.config
def run_config():    
    train = True # Train or play?
    hidden_size = 32
    epsilon = 0.0

    # Train specific
    lr = 0.005
    batch_size = 1024
    solved_min_reward = 200 # Solved when min reward is at least this
    solved_n_episodes =  100 # for over this many episodes
    max_steps = 10**7
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_warmup_episodes = 30
    n_episodes_per_iter = 10
    n_updates_per_iter = 50
    eval_episodes = 100
    eval_every_n_steps = 50_000
    max_return = 300

    experiment_name = f'lunarlander_hs{hidden_size}_mr{max_return}_b{batch_size}_rs{replay_size}_lf{last_few}_nw{n_warmup_episodes}_ne{n_episodes_per_iter}_nu{n_updates_per_iter}_e{epsilon}_ev{eval_episodes}'
    checkpoint_path = f'checkpoint_{experiment_name}.pt'


    # Play specific
    sample_action = True
    play_episodes = 5
    dh = 200
    dr = 400



@ex.automain
def main():
    """
    Default runs train() command
    """
    train()