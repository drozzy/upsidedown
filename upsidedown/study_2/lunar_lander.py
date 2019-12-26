import gym
import os
import numpy as np
import random
import torch
from torch import nn
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from experiment import rollout, ReplayBuffer, Trajectory, load_model, save_model
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment()

MODEL_NAME = 'model_v0_lunar_lander_v2'

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

@ex.capture
def run_train(batch_size, solved_mean_reward, max_steps, hidden_size, replay_size, last_few, n_warmup_episodes,
    n_episodes_per_iter, n_updates_per_iter, start_epsilon, eval_every):
    writer = SummaryWriter()
    env = gym.make('LunarLander-v2')
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(state_shape=env.observation_space.shape[0], cmd_shape=2, num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)

    n_warmup_episodes = 30
    # Random rollout
    trajectories, mean_reward, length = rollout(episodes=n_warmup_episodes, env=env, render=False)
    rb.add(trajectories)

    # Keep track of steps used during random rollout!
    epoch, model, optimizer, loss, steps = load_model(MODEL_NAME, model, optimizer, device, train=True)
    steps += length
    save_model(MODEL_NAME, epoch, model, optimizer, loss, steps)

    # Plot initial values
    writer.add_scalar('Steps/reward', mean_reward, steps)     

    loss_sum = 0
    loss_count = 0
    rewards = []

    for i in count(start=epoch):
        x, y = rb.sample(batch_size, device)    
        loss = train_step(model, x, y)
        loss_sum += loss
        loss_count += 1
        
        writer.add_scalar('Loss/loss', loss, i)
        
        (dh, dr) = rb.sample_command()
        writer.add_scalar('Epoch/dh', dh, i)
        writer.add_scalar('Epoch/dr', dr, i)

        if i % n_updates_per_iter == 0:
            trajectories, mean_reward, length = rollout(n_episodes_per_iter, env=env, model=model, 
                sample_action=True, replay_buffer=rb, device=device, action_fn=action_fn, 
                epsilon=start_epsilon)
            rb.add(trajectories)
            rewards.append(mean_reward)
            rewards = rewards[-50:] # Keep only last  rewards
            
            steps += length
            avg_loss = loss_sum/loss_count
            save_model(MODEL_NAME, i, model, optimizer, avg_loss, steps)        
            print(f"Average Episode Reward: {mean_reward}")        
            writer.add_scalar('Steps/reward', mean_reward, steps)
            
            mean_length = length*1.0/n_episodes_per_iter
            writer.add_scalar('Steps/length', mean_length, steps)
            
            if np.mean(rewards) >= solved_mean_reward:
                print("Task considered solved! Stopping.")
                break
            
            if steps >= max_steps:
                print(f"Steps {steps} exceeds max env steps {MAX_STEPS}. Stopping.")
                break
                
        if i % eval_every == 0:
            eval_episodes = 10
            _, mean_reward, length = rollout(eval_episodes, env=env, model=model, 
                    sample_action=True, replay_buffer=rb, 
                    device=device, action_fn=action_fn, evaluation=True)

            writer.add_scalar('Eval/reward', mean_reward, i)        
            mean_length = length*1.0/n_episodes_per_iter
            writer.add_scalar('Eval/length', mean_length, i)
            
        if i % 200 == 0:
            avg_loss = loss_sum/loss_count
            print(f'i: {i}, s: {steps}, Loss: {avg_loss}')
            
            save_model(MODEL_NAME, i, model, optimizer, avg_loss, steps)
           
def train_step(inputs, targets, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model([inputs[:, :-2], inputs[:, -2:]])
    loss = loss_object(predictions, targets)
    
    loss.backward()
    optimizer.step()
    
    return loss

def action_fn(env, model, inputs, sample_action, epsilon):
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

@ex.capture
def run_play(epsilon, sample_action, hidden_size, episodes):
    env = gym.make('LunarLander-v2')
    cmd = (200, 200)
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape[0], cmd_shape=2, num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # dh ,dr
    # cmd = rb.sample_command()
    # rb.sample_command()
    env = gym.make('LunarLander-v2')
    e, model, _, l,_ = load_model(name=MODEL_NAME, train=False, 
        model=model, optimizer=optimizer, device=device)

    for _ in range(episodes):
        _, mean_reward, _ = rollout(episodes=1, env=env, model=model, sample_action=sample_action, 
                              cmd=cmd, render=True, device=device, action_fn=action_fn)

        print(f"Episode Reward: {mean_reward}")


@ex.config
def run_play_config():
    epsilon = 0.0
    sample_action = True
    episodes = 5
    
@ex.config
def run_train_config():
    batch_size = 1024
    solved_mean_reward = 200 #
    max_steps = 10**7
    hidden_size = 32
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_warmup_episodes = 10
    n_episodes_per_iter = 10
    n_updates_per_iter = 50
    start_epsilon = 0.1
    eval_every = 1000

@ex.automain
def main(train):
    if train:
        run_train()
    else:
        run_play()