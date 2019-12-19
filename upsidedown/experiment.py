import numpy as np

def rollout_random(num_episodes, env, replay_buffer, render=False):
    avg_rewards = []
    for _ in range(num_episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        t = Trajectory()
        while not done:            
            s_old = s
            action = env.action_space.sample()
            if render:
                env.render()

            s, reward, done, info = env.step(action)
            t.add(s_old, action, reward, s)
            ep_reward += reward
        avg_rewards.append(ep_reward)    
        replay_buffer.add(t)

    if render:
        env.close()

    return np.mean(avg_rewards)

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
        
    def add(self, trajectory):
        self.buffer.append(trajectory)
        
        self.buffer = sorted(self.buffer, key=lambda x: x.total_return, reverse=True)
        self.buffer = self.buffer[:self.max_size]
        
    def sample(self, batch_size):
        trajectories = np.random.choice(self.buffer, batch_size, replace=True)
        
        segments = []
        
        for t in trajectories:
            segments.append(t.sample_segment())
            
        return segments
    
    def sample_command(self):
        eps = self.buffer[:self.last_few]
        
        dh_0 = np.mean([e.length for e in eps])
        
        m = np.mean([e.total_return for e in eps])
        s = np.std([e.total_return for e in eps])
        
        dr_0 = np.random.uniform(m, m+s)
        
        return dh_0, dr_0
