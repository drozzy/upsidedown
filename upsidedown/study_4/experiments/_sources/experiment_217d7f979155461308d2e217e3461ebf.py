from collections import namedtuple
import torch.nn.functional as F
import time 
import gym
from gym.wrappers.time_limit import TimeLimit
from itertools import cycle
import os
import numpy as np
import random
import datetime
import torch
from torch import nn
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from sacred import Experiment
device = torch.device("cpu")

ex = Experiment()

# Segment from a trajectory (numpy, single elements)
Segment = namedtuple('Segment', ['prev_action', 'state', 'dr', 'action', 'episode_over'])

# Batch-sized sample from the replay buffer, each element already a torch.tensor on device
Sample = namedtuple('Sample', ['prev_action', 'state', 'dr', 'action', 'episode_over'])

# Command expressing desired quantities (numpy, single elements) 
Command = namedtuple('Command', ['dr', 'episode_over'])
           
def get_action(env, model, prev_action, state, cmd, sample_action, epsilon, device):
    prev_action = torch.tensor(prev_action).long().unsqueeze(dim=0).to(device)
    episode_over = torch.tensor(cmd.episode_over).long().unsqueeze(dim=0).to(device)
    dr = torch.tensor([cmd.dr]).float().unsqueeze(dim=0).to(device)
    state = torch.tensor(state).float().unsqueeze(dim=0).to(device)
    action_logits = model(prev_action=prev_action, state=state, dr=dr, episode_over=episode_over)
    action_probs = torch.softmax(action_logits, axis=-1)

    if random.random() < epsilon: # Random action
        return env.action_space.sample()
    
    if sample_action:        
        m = torch.distributions.categorical.Categorical(logits=action_logits)             
        action = int(m.sample().squeeze().cpu().numpy())        
    else:
        action = int(np.argmax(action_probs.detach().squeeze().numpy()))
    return action

def rollout_episode(env, model, sample_action, cmd, 
                    render, device, epsilon, max_return):
    s = env.reset()
    done = False
    ep_reward = 0.0
    prev_action = env.action_space.n # First action is "special" start of episode action

    t = Trajectory()
    
    while not done:
        with torch.no_grad():
            model.eval()
            action = get_action(env, model, prev_action=prev_action, state=s, cmd=cmd, sample_action=sample_action, epsilon=epsilon, device=device)
            model.train()
        
        if render:
            env.render()
            time.sleep(0.01)
            
        s_old = s        
        s, reward, done, info = env.step(action)
        
        dr = np.clip(cmd.dr - reward, -max_return, max_return)
        cmd = Command(dr=dr, episode_over=cmd.episode_over)
            
        t.add(prev_action, s_old, action, reward, s)
        prev_action = action    
        ep_reward += reward

    return t, ep_reward

def rollout(episodes, env, epsilon, max_return, model, cmd, sample_action=True, render=False, 
            replay_buffer=None, device=None):
    """
    @param model: Model to use to select an action.
    @param sample_action=True: If True samples action from distribution, otherwise 
                                selects max.
    @param epsilon - Probability of doing a random action. Between 0 and 1.0. Or -1 if no random actions are needed?
    """
    assert cmd is not None
    assert model is not None

    trajectories = []
    rewards = [] 
    length = 0

    for e in range(episodes):
        t, reward = rollout_episode(env=env, model=model, sample_action=sample_action, cmd=cmd,
                            render=render, device=device, epsilon=epsilon, max_return=max_return)            
        
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

@ex.capture
def get_checkpoint_path(_run, experiment_name, existing_checkpoint_path):
    if existing_checkpoint_path is not None:
        return existing_checkpoint_path

    run_id = _run._id or datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    return f'checkpoint_{run_id}_{experiment_name}.pt'

def save_checkpoint(model, optimizer, loss, updates, steps):    
    checkpoint_path = get_checkpoint_path()

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'updates': updates,
            'steps': steps}, 
            checkpoint_path)
    
def load_checkpoint(model, optimizer, device, train=True):
    checkpoint_path = get_checkpoint_path()

    loss = 0.0    
    steps = 0
    updates = 0
    if os.path.exists(checkpoint_path):        
        checkpoint = torch.load(checkpoint_path)
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
        # t1 = int(random.random() * (T+1)) # THIS IS FASTER by about 1/4
        t2 = random.randint(t1, T)
        
        prev_action = self.trajectory[t1-1][0]
        state = self.trajectory[t1-1][1]
        action = self.trajectory[t1-1][2]
        episode_over = 1 if (t2 == T) else 0 # TODO
        d_r = self.cum_sum[t2 - 1] - self.cum_sum[t1 - 2]

        return Segment(prev_action=prev_action, state=state, dr=d_r, action=action, episode_over=episode_over)

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

    def clear(self):
        self.cur_size = 0
        self.buffer = [] 
            
    def _add(self, trajectory):
        self.buffer.append(trajectory)
        
        self.buffer = sorted(self.buffer, key=lambda x: x.total_return, reverse=True)
        self.buffer = self.buffer[:self.max_size]
    
    def sample(self, batch_size, device):
        trajectories = np.random.choice(self.buffer, batch_size, replace=True)
        prev_action_batch = []
        action_batch = []
        s_batch  = []
        dr_batch = []
        episode_over_batch = []

        for t in trajectories:
            
            segment = t.sample_segment()

            s_batch.append(segment.state)
            dr_batch.append(segment.dr)
            episode_over_batch.append(segment.episode_over)
            prev_action_batch.append(segment.prev_action)
            
            action_batch.append(segment.action)
            if len(s_batch) >= batch_size:
                break

        s_batch = torch.tensor(s_batch).float().to(device)
        dr_batch = torch.tensor(dr_batch).unsqueeze(dim=1).to(device)
        episode_over_batch = torch.tensor(episode_over_batch).long().to(device)
        prev_action_batch = torch.tensor(prev_action_batch).long().to(device)

        action_batch = torch.tensor(action_batch).to(device)

        return Sample(prev_action=prev_action_batch, state=s_batch, dr=dr_batch, action=action_batch, episode_over=episode_over_batch)
    
    @ex.capture
    def sample_command(self, dr, episode_over):
        # Special case: when there is no experience yet
        if len(self.buffer) == 0:
            return Command(dr=dr, episode_over=episode_over)

        episodes = self.buffer[:self.last_few]
        
        # This seems to work for cartpole:
        # max_return = np.max([e.total_return for e in episodes])
        # dr_0 = max(2 * max_return, 1)

        # This seems to work for lunar-lander
        # Pick randomly a target flag
        episode_over = 0
        m = np.mean([e.total_return for e in episodes])
        s = np.std([e.total_return for e in episodes])        
        dr_0 = np.random.uniform(m, m + s)

        return Command(episode_over=episode_over, dr=dr_0)

    def eval_command(self):
        episodes = self.buffer[:self.last_few]
        
        episode_over = 0
        
        dr_0 = np.min([e.total_return for e in episodes])
        
        return Command(episode_over=episode_over, dr=dr_0)


def swish(x, beta=1):                                                                                                                                                                                      
    return x * torch.sigmoid(beta * x)

class Behavior(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, state_shape, num_actions, return_scale, horizon_scale):
        super(Behavior, self).__init__()
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        # Extra action representation for "start" of episode action.
        self.emb_prev_action = torch.nn.Embedding(num_embeddings=num_actions+1, embedding_dim=hidden_size)
        self.emb_episode_over = torch.nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)
        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_attn_scores = nn.Linear(state_shape, state_shape)
        self.fc_dr = nn.Linear(1, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, prev_action, state, dr, episode_over):
        output_prev_action = self.emb_prev_action(prev_action)
        output_episode_over = self.emb_episode_over(episode_over)
        attn_scores = self.fc_attn_scores(state)
        output_state = self.fc_state(torch.softmax(attn_scores, dim=1) * state)
        output_dr = torch.sigmoid(self.fc_dr(dr * self.return_scale))

        output = output_prev_action * output_state * output_dr * output_episode_over
        
        # sum1 = (output_prev_action + output_state)
        # output = sum1 * sum2 # TODO: Is this a good way to combine these?
        
        output = swish(self.fc1(output))
        output = self.fc2(output)
        return output



@ex.command
def train(env_name, _run, experiment_name, hidden_size, lr, replay_size, last_few, max_episode_steps):
    """
    Begin or resume training a policy.
    """
    run_id = _run._id or datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_dir = f'tensorboard/{run_id}_{experiment_name}'
    checkpoint_path = get_checkpoint_path()
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(env_name)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    loss_object = torch.nn.CrossEntropyLoss().to(device)
    
    model = Behavior(hidden_size=hidden_size, state_shape=env.observation_space.shape[0], num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    

    print("Trying to load:")
    print(checkpoint_path)
    c = load_checkpoint(model, optimizer, device, train=True)
    updates, steps, loss = c.updates, c.steps, c.loss

    last_eval_step = 0
    rewards = []
    done = False

    rb = ReplayBuffer(max_size=replay_size, last_few=last_few)
    while not done:
        steps, updates, last_eval_step, done = do_iteration(env=env, model=model, rb=rb, optimizer=optimizer, 
            loss_object=loss_object,  writer=writer, updates=updates, steps=steps,
            last_eval_step=last_eval_step, rewards=rewards)

    add_artifact()

@ex.capture
def do_iteration(env, model, optimizer, loss_object, writer, updates, steps, last_eval_step, rewards, rb):
    # Exloration
    print("Beginning exploration.")
    steps = do_exploration(env, model, rb, writer, steps)
    
    # Updates    
    print("Beginning updates.")
    updates = do_updates(model, optimizer, loss_object, rb, writer, updates, steps)
        
    # Evaluation
    print("Beginning evaluation.")
    last_eval_step, done = do_eval(env=env, model=model, rb=rb, writer=writer, steps=steps, 
        rewards=rewards, last_eval_step=last_eval_step)

    return steps, updates, last_eval_step, done

@ex.capture
def do_eval(env, model, rb, writer, steps, rewards, last_eval_step, eval_episodes, 
    max_return, max_steps, solved_min_reward, solved_n_episodes, eval_every_n_steps):
    
    eval_cmd = rb.eval_command()

    roll = rollout(eval_episodes, env=env, model=model, epsilon=0.0,
            sample_action=True, replay_buffer=rb, 
            device=device, cmd=eval_cmd,
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
    
    done = solved or steps_exceeded
    return last_eval_step, done
 

@ex.capture
def do_updates(model, optimizer, loss_object, rb, writer, updates, steps, 
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

    save_checkpoint(model=model, optimizer=optimizer, loss=avg_loss, updates=updates, steps=steps)
    return updates


@ex.capture
def do_exploration(env, model, rb, writer, steps, n_episodes_per_iter, epsilon, max_return):
    exploration_cmd = rb.sample_command()

    # NOW CLEAR the buffer
    rb.clear()

    writer.add_scalar('Exploration/dr', exploration_cmd.dr, steps)

    # Exploration    
    print("Beggining rollout.")
    roll = rollout(n_episodes_per_iter, env=env, model=model, cmd=exploration_cmd,
        sample_action=True, replay_buffer=rb, device=device, 
        epsilon=epsilon, max_return=max_return)
    rb.add(roll.trajectories)
    print("End rollout.")

    steps += roll.length
    
    writer.add_scalar('Exploration/reward', roll.mean_reward, steps)
    writer.add_scalar('Exploration/length', roll.mean_length, steps)

    return steps

def add_artifact():
    checkpoint_path = get_checkpoint_path()
    ex.add_artifact(checkpoint_path, name='checkpoint.pt')
           
def train_step(sample, model, optimizer, loss_object):
    optimizer.zero_grad()    
    predictions = model(prev_action=sample.prev_action, state=sample.state, dr=sample.dr, episode_over=sample.episode_over)
    loss = loss_object(predictions, sample.action)

    loss.backward()
    optimizer.step()
    
    return loss


@ex.command
def play(env_name, sample_action, max_return, hidden_size, play_episodes, episode_over, dr, max_episode_steps):
    """
    Play episodes using a trained policy. 
    """
    env = gym.make(env_name)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    cmd = Command(dr=dr, episode_over=episode_over)

    loss_object = torch.nn.CrossEntropyLoss().to(device)
    model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape[0], num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    c = load_checkpoint(train=False, 
        model=model, optimizer=optimizer, device=device)

    for _ in range(play_episodes):
        roll = rollout(episodes=1, env=env, epsilon=0.0, max_return=max_return, model=model, sample_action=sample_action, 
                              cmd=cmd, render=True, device=device)

        print(f"Episode Reward: {roll.mean_reward}, episode length: {roll.length}")


@ex.config
def run_config():    
    # Environment to train on
    env_name = 'LunarLander-v2'
    max_episode_steps = None # If set, stop the episode after this many steps

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
    eval_episodes = 10
    eval_every_n_steps = 50_000
    max_return = 400 # Maximum possible return for this environment (absolute value)
    dr = 400
    episode_over = 0

    experiment_name = f'{env_name.replace("-", "_")}_hs{hidden_size}_mr{max_return}_b{batch_size}_rs{replay_size}_lf{last_few}_ne{n_episodes_per_iter}_nu{n_updates_per_iter}_e{epsilon}_lr{lr}'
    
    existing_checkpoint_path = None


    # Play specific
    sample_action = True
    play_episodes = 5



@ex.automain
def main():
    train()