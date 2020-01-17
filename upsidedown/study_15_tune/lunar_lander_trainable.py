from ray.tune import Trainable
from lib import ReplayBuffer, Command

class LunarLanderTrainable(Trainable):
    def _setup(self, config):
        self.config = config
        self.config.update(self.default_config)

        # Fill in from config
        self.env_name = config['env_name']
        self.num_stack = config['num_stack']
        self.hidden_size = config['hidden_size']
        self.lr = config['lr']
        self.replay_size = config['replay_size']
        self.last_few = config['last_few']
        self.n_episodes_per_iter = config['n_episodes_per_iter']
        self.epsilon = config['epsilon']
        self.max_return = config['max_return']
        self.episodes = config['episodes']
        self.render = config['render']

        # Initialize 
        self.device = torch.device("cpu")
        self.steps = 0
        self.updates = 0
        self.loss = None

        self.env =  FrameStack(gym.make(self.env_name), num_stack=self.num_stack)
        self.loss_object = torch.nn.CrossEntropyLoss().to(device)

        self.model = Behavior(hidden_size=self.hidden_size, state_shape=self.env.observation_space.shape, num_actions=self.env.action_space.n).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.rb = ReplayBuffer(max_size=self.replay_size, last_few=self.last_few)
        # env_name, _run, experiment_name, hidden_size, lr, replay_size, last_few, max_episode_steps, num_stack):

    

    def play(self, dr=300, dh=300, sample_action=True, play_episodes=5):
        """
        Play a few episodes with the currently trained policy.
        """
        cmd = Command(dr=dr, dh=dh)

        # loss_object = torch.nn.CrossEntropyLoss().to(device)
        # model = Behavior(hidden_size=hidden_size,state_shape=env.observation_space.shape, num_actions=env.action_space.n).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        for _ in range(play_episodes):
            roll = self.rollout(episodes=1, sample_action=sample_action, 
                                  cmd=cmd, render=True)

            print(f"Episode Reward: {roll.mean_reward} Steps: {roll.length}")

    def _train(self):
        """
        Do one iteration of training
        """
        # last_eval_step = 0
        # rewards = []
        # done = False
        # while not done:
        #     steps, updates, last_eval_step, done = do_iteration(env=env, model=model, rb=rb, optimizer=optimizer, 
        #         loss_object=loss_object,  writer=writer, updates=updates, steps=steps,
        #         last_eval_step=last_eval_step, rewards=rewards)

        return self.do_iteration()

        # add_artifact()

    def do_iteration(env, model, optimizer, loss_object, writer, updates, steps, last_eval_step, rewards, rb):
        results = {}

        # Exloration
        print("Beginning exploration.")
        dr, dh, steps, mean_reward, mean_length = self.do_exploration()
        
        results['Rollout_Exploration/dr'] = dr
        results['Rollout_Exploration/dh'] = dh
        results['Rollout_Exploration/reward_mean'] = mean_reward
        results['Rollout_Exploration/length_mean'] = mean_length
        results['Rollout_Exploration/steps'] = steps
        results['timesteps_this_iter'] = steps
        
        # TODO: uncomment.
        # # Updates    
        # print("Beginning updates.")
        # updates = do_updates(model, optimizer, loss_object, rb, writer, updates, steps)
            
        # # Evaluation
        # print("Beginning evaluation.")
        # last_eval_step, done = do_eval(env=env, model=model, rb=rb, writer=writer, steps=steps, 
        #     rewards=rewards, last_eval_step=last_eval_step)

        mean, std, mean_last, std_last, mean_len, std_len, mean_len_last, std_len_last = self.rb.stats()
        
        results['Buffer_Rewards/mean'] = mean
        results['Buffer_Rewards/std'] = std
        results['Buffer_Rewards/mean_last_few'] = mean_last
        results['Buffer_Rewards/std_last_few'] = std_last

        results['Buffer_Lengths/mean'] = mean_len
        results['Buffer_Lengths/std'] = std_len
        results['Buffer_Lengths/mean_last_few'] = mean_len_last
        results['Buffer_Lengths/std_last_few'] = std_len_last

        # TODO: return also
        # episode_reward_mean
        # mean_loss
        # mean_accuracy
        

        return results # steps, updates, last_eval_step, done   

    def do_exploration(self):
        # Sample command for the exploration
        exploration_cmd = self.rb.sample_command(self.init_dr, self.init_dh)

        # NOW CLEAR the buffer
        rb.clear()

        # Exploration    
        roll = self.rollout(cmd=exploration_cmd, sample_action=True)
        self.rb.add(roll.trajectories)

        steps = roll.length
        return dr, dh, roll.length, roll.mean_reward, roll.mean_length

    def rollout(self, sample_action=True, cmd=None, render=False):
        assert cmd is not None

        trajectories = []
        rewards = [] 
        length = 0

        for e in range(self.episodes):
            t, reward = rollout_episode(sample_action=sample_action, cmd=cmd, render=render, epsilon=epsilon)
            
            trajectories.append(t)
            length += t.length
            rewards.append(reward)
        
        if render:
            self.env.close()
        
        return Rollout(episodes=self.episodes, trajectories=trajectories, rewards=rewards, length=length)

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
            
            if model is not None:
                dh = max(cmd.dh - 1, 1)
                dr = min(cmd.dr - reward, max_return)
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
            self.loss = checkpoint['loss']
            self.updates = checkpoint['updates']
            self.steps = checkpoint['steps']

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'replay_buffer' : self.rb.state_dict(),
                'loss': self.loss,
                'updates': self.updates,
                'steps': self.steps}, 
                checkpoint_path)
        return checkpoint_path

    @param
    def default_config(self):
        # Environment to train on
        return {
            'env_name': 'LunarLander-v2',
            'num_stack' : 10,
            'hidden_size' : 32,
            'epsilon' : 0.0,
            'return_scale': 0.01,
            'horizon_scale' : 0.001,
            'lr' = 0.005
            'batch_size' : 1024,
            # Solved when min reward is at least this ...
            'solved_min_reward' : 200,
            # ... over this many episodes
            'solved_n_episodes' :  100,
            'max_steps' : 10**7,
            # Maximum size of the replay buffer in episodes
            'replay_size' : 100,
            'n_episodes_per_iter' : 100,
            'last_few' : 10,
            'n_updates_per_iter' : 200,
            'eval_episodes' : 10,
            'eval_every_n_steps' : 5_000,
            'max_return' : 300
            # Initial dh, dr values to use when our buffer is empty
            'init_dh' : 1,
            'ini_dr' : 0
        }

# Other implementation methods that may be helpful to override are _log_result, reset_config, _stop, and _export_model.