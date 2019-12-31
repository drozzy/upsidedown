from collections import namedtuple
import torch.nn.functional as F
import time 
import gym
import os
import numpy as np
import random
import datetime
import torch
from torch import nn
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from sacred import Experiment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment()
           
def get_action(env, model, prev_action, state, cmd, sample_action, epsilon, device):
    prev_action = torch.tensor(prev_action).long().unsqueeze(dim=0).to(device)
    dr = torch.tensor(cmd.dr).float().unsqueeze(dim=0).to(device)
    dh = torch.tensor(cmd.dh).float().unsqueeze(dim=0).to(device)
    state = torch.tensor(state).float().unsqueeze(dim=0).to(device)

    action_logits = model(prev_action=prev_action, state=state, dr=dr, dh=dh)
    action_probs = torch.softmax(action_logits, axis=-1)

    if random.random() < epsilon: # Random action
        return env.action_space.sample()
    
    if sample_action:        
        m = torch.distributions.categorical.Categorical(logits=action_logits)             
        action = int(m.sample().squeeze().cpu().numpy())        
    else:
        action = int(np.argmax(action_probs.detach().squeeze().numpy()))
    return action

Command = namedtuple('Command', ['dr', 'dh'])

def rollout_episode(env, model, sample_action, cmd, 
                    render, device, epsilon, max_return=300):
    s = env.reset()
    done = False
    ep_reward = 0.0
    prev_action = env.action_space.n # First action is "special" start of episode action

    t = Trajectory()
    
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:            
            # (dh, dr) = cmd
            # (s, dr, dh), action = segment
            # # l = to_training(s, dr, dh)
            # s_batch.append(s)
            # dr_batch.append(dr)
            # dh_batch.append(dh)

            # inputs = torch.tensor([to_training(s, dr, dh)]).float().to(device)
            with torch.no_grad():
                model.eval()
                action = get_action(env, model, prev_action=prev_action, state=s, cmd=cmd, sample_action=sample_action, epsilon=epsilon, device=device)
                model.train()
        
        if render:
            env.render()
            time.sleep(0.01)
            
        s_old = s        
        s, reward, done, info = env.step(action)
        
        if model is not None:
            dh = max(cmd.dh - 1, 1)
            dr = min(cmd.dr - reward, max_return)
            cmd = Command(dr=dr, dh=dh)
            # cmd = (dh, dr)
            
        t.add(prev_action, s_old, action, reward, s)    
        prev_action = action    
        ep_reward += reward
    
    
    return t, ep_reward

def rollout(episodes, env, model=None, sample_action=True, cmd=None, render=False, 
            replay_buffer=None, device=None, evaluation=False, epsilon=0.0, max_return=300):
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
        print(f"Existing model found. Loading from steps: {steps}, updates: {updates}, with loss: {loss}")
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
        
    def add(self, prev_action, state, action, reward, state_prime):
        self.trajectory.append((prev_action, state, action, reward, state_prime))
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
        
        prev_action = self.trajectory[t1-1][0]
        state = self.trajectory[t1-1][1]
        action = self.trajectory[t1-1][2]

        d_r = self.cum_sum[t2 - 1] - self.cum_sum[t1 - 2]
        
        d_h = t2 - t1 + 1.0

        return Segment(prev_action=prev_action, state=state, dr=d_r, dh=d_h, action=action)

Segment = namedtuple('Segment', ['prev_action', 'state', 'dr', 'dh', 'action'])

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
        
    def add(self, trajectories):
        for trajectory in trajectories:
            self._add(trajectory)        
            
    def _add(self, trajectory):
        self.buffer.append(trajectory)
        
        self.buffer = sorted(self.buffer, key=lambda x: x.total_return, reverse=True)
        self.buffer = self.buffer[:self.max_size]
    
    def sample(self, batch_size, device):
        trajectories = np.random.choice(self.buffer, batch_size, replace=True)
        # x = []
        prev_action_batch = []
        action_batch = []
        s_batch  = []
        dr_batch = []
        dh_batch = []
        for t in trajectories:
            
            segment = t.sample_segment()
            # prev_action, (s, dr, dh), action = segment
            # l = to_training(s, dr, dh)
            s_batch.append(segment.state)
            dr_batch.append(segment.dr)
            dh_batch.append(segment.dh)
            prev_action_batch.append(segment.prev_action)
            # l = s.tolist()
            # l.append(dh*horizon_scale)
            # l.append(dr*return_scale)
            
            action_batch.append(segment.action)

        s_batch = torch.tensor(s_batch).to(device)
        dr_batch = torch.tensor(dr_batch).unsqueeze(dim=1).to(device)
        dh_batch = torch.tensor(dh_batch).unsqueeze(dim=1).to(device)
        prev_action_batch = torch.tensor(prev_action_batch).long().to(device)

        # x.append(l)
            
        # x = torch.tensor(x).to(device)
        action_batch = torch.tensor(action_batch).to(device)

        return Sample(prev_action=prev_action_batch, state=s_batch, dr=dr_batch, dh=dh_batch, action=action_batch)
    
    def sample_command(self):
        # Special case: when there is no experience yet
        if len(self.buffer) == 0:
            return Command(dr=0, dh=0)

        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        m = np.mean([e.total_return for e in eps])
        s = np.std([e.total_return for e in eps])
#         s = median_absolute_deviation([e.total_return for e in eps])
        
        dr_0 = np.random.uniform(m, m + s)
        
        return Command(dh=dh_0, dr=dr_0)

    def eval_command(self):
        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        dr_0 = np.min([e.total_return for e in eps])
        
        return Command(dh=dh_0, dr=dr_0)


class Behavior(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, state_shape, num_actions, return_scale, horizon_scale):
        super(Behavior, self).__init__()
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        # Extra action representation for "start" of episode action.
        self.emb_prev_action = torch.nn.Embedding(num_embeddings=num_actions+1, embedding_dim=hidden_size)
        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_dr = nn.Linear(1, hidden_size)
        self.fc_dh = nn.Linear(1, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, prev_action, state, dr, dh):
        output_prev_action = self.emb_prev_action(prev_action)
        output_state = self.fc_state(state)
        output_dr = torch.sigmoid(self.fc_dr(dr * self.return_scale))
        output_dh = torch.sigmoid(self.fc_dh(dh * self.horizon_scale))
        
        sum1 = (output_prev_action + output_state)
        sum2 = (output_dr + output_dh)
        output = sum1 * sum2 # TODO: Is this a good way to combine these?
        
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output

@ex.command
def train(_run, experiment_name, hidden_size, replay_size, last_few, lr, checkpoint_path):
    """
    Begin or resume training a policy.
    """
    run_id = _run._id or datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_dir = f'tensorboard/{run_id}_{experiment_name}'
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make('LunarLander-v2')
    
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    
    model = Behavior(hidden_size=hidden_size, state_shape=env.observation_space.shape[0], num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)

    print("Trying to load:")
    print(checkpoint_path)
    c = load_checkpoint(checkpoint_path, model, optimizer, device, train=True)
    updates, steps, loss = c.updates, c.steps, c.loss

    last_eval_step = 0
    rewards = []
    done = False

    while not done:
        steps, updates, last_eval_step, done = do_iteration(env=env, model=model, optimizer=optimizer, 
            loss_object=loss_object, rb=rb, writer=writer, updates=updates, steps=steps,
            last_eval_step=last_eval_step, rewards=rewards)

    add_artifact()

@ex.capture
def do_iteration(env, model, optimizer, loss_object, rb, writer, updates, steps, last_eval_step, rewards):

    # Exloration
    steps = do_exploration(env, model, rb, writer, steps)
    
    # Updates    
    updates = do_updates(model, optimizer, loss_object, rb, writer, updates, steps)
        
    # Evaluation
    last_eval_step, done = do_eval(env=env, model=model, rb=rb, writer=writer, steps=steps, 
        rewards=rewards, last_eval_step=last_eval_step)

    return steps, updates, last_eval_step, done

@ex.capture
def do_eval(env, model, rb, writer, steps, rewards, last_eval_step, eval_episodes, 
    max_return, max_steps, solved_min_reward, solved_n_episodes, eval_every_n_steps):

    roll = rollout(eval_episodes, env=env, model=model, 
            sample_action=True, replay_buffer=rb, 
            device=device, evaluation=True,
            max_return=max_return)

    steps_exceeded = steps >= max_steps
    time_to_eval = ((steps - last_eval_step) >= eval_every_n_steps) or steps_exceeded or (last_eval_step == 0)

    if steps_exceeded:
        print(f"Steps {steps} exceeds max env steps {max_steps}.")

    if not time_to_eval:
        return last_eval_step, steps_exceeded

    last_eval_step = steps

    cmd = rb.eval_command()
    writer.add_scalar('Eval/dr', cmd.dr, steps)
    writer.add_scalar('Eval/dh', cmd.dh, steps)
    
    writer.add_scalar('Eval/reward', roll.mean_reward, steps) 
    writer.add_scalar('Eval/length', roll.mean_length, steps)
    
    print(f"Eval Episode Mean Reward: {roll.mean_reward}")      

    # Stopping criteria
    rewards.extend(roll.rewards)
    rewards = rewards[-solved_n_episodes:]
    eval_min_reward = np.min(rewards)

    solved = eval_min_reward >= solved_min_reward
    if solved:
        print(f"Task considered solved. Achieved {eval_min_reward} >= {solved_min_reward} over {solved_n_episodes} episodes.")
        
    return last_eval_step, solved
 

@ex.capture
def do_updates(model, optimizer, loss_object, rb, writer, updates, steps, checkpoint_path, 
    batch_size, n_updates_per_iter):
    loss_sum = 0
    loss_count = 0
   
    
    for _ in range(n_updates_per_iter):
        updates +=1

        sample = rb.sample(batch_size, device)    

        loss = train_step(sample, model, optimizer, loss_object)
        loss_sum += loss
        loss_count += 1            

    # Save updated model
    avg_loss = loss_sum/loss_count
    print(f'u: {updates}, s: {steps}, Loss: {avg_loss}')
    writer.add_scalar('Loss/avg_loss', avg_loss, steps)

    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, loss=avg_loss, updates=updates, steps=steps)
    return updates


@ex.capture
def do_exploration(env, model, rb, writer, steps, n_episodes_per_iter, epsilon, max_return):
    # Plot a sample dr/dh at this time
    example_cmd = rb.sample_command()

    writer.add_scalar('Exploration/dr', example_cmd.dr, steps)
    writer.add_scalar('Exploration/dh', example_cmd.dh, steps)

    # Exploration    
    roll = rollout(n_episodes_per_iter, env=env, model=model, 
        sample_action=True, replay_buffer=rb, device=device, 
        epsilon=epsilon, max_return=max_return)
    rb.add(roll.trajectories)

    steps += roll.length
    
    writer.add_scalar('Exploration/reward', roll.mean_reward, steps)
    writer.add_scalar('Exploration/length', roll.mean_length, steps)

    return steps

@ex.capture
def add_artifact(checkpoint_path):
    ex.add_artifact(checkpoint_path, name='checkpoint.pt')
           
def train_step(sample, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model(prev_action=sample.prev_action, state=sample.state, dr=sample.dr, dh=sample.dh)
    loss = loss_object(predictions, sample.action)
    
    loss.backward()
    optimizer.step()
    
    return loss


@ex.command
def play(checkpoint_path, epsilon, sample_action, hidden_size, play_episodes, dh, dr):
    """
    Play episodes using a trained policy. 
    """
    env = gym.make('LunarLander-v2')
    cmd = Command(dr=dr, dh=dh)

    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape[0], num_actions=env.action_space.n).to(device)
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
    epsilon = 0.1
    return_scale = 0.01
    horizon_scale = 0.01

    # Train specific
    lr = 0.005
    batch_size = 1024
    solved_min_reward = 200 # Solved when min reward is at least this
    solved_n_episodes =  100 # for over this many episodes
    max_steps = 10**7
    replay_size = 100 # Maximum size of the replay buffer in episodes
    last_few = 50     
    n_episodes_per_iter = 10
    n_updates_per_iter = 50
    eval_episodes = 100
    eval_every_n_steps = 50_000
    max_return = 300

    experiment_name = f'lunarlander_hs{hidden_size}_mr{max_return}_b{batch_size}_rs{replay_size}_lf{last_few}_ne{n_episodes_per_iter}_nu{n_updates_per_iter}_e{epsilon}_lr{lr}'
    checkpoint_path = f'checkpoint_{experiment_name}.pt'


    # Play specific
    sample_action = True
    play_episodes = 5
    dh = 200
    dr = 400



@ex.automain
def main():
    train()