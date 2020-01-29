from lunar_lander_trainable import LunarLanderTrainable
import ray
from ray import tune
# from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers import ASHAScheduler
import random
import numpy as np

def main():
    ray.init(num_cpus=1)
    # To debug in sequential mode run:
    # ray.init(local_mode=True)
    # analysis = tune.run(
    #     LunarLanderTrainable,
    #     config={
    #         'max_steps' : 10**3
    #     },
    #     # Repeat experiments multiple times
    #     num_samples=10,
    #     checkpoint_freq=1,
    #     checkpoint_at_end=True,
    #     max_failures=0
    # )

    # pbt_scheduler = PopulationBasedTraining(
    #     time_attr='training_iteration',
    #     metric='episode_reward_mean',
    #     mode='max',
    #     perturbation_interval=5, # 600.0,
    #     hyperparam_mutations={
    #         # Prob of this mutation being performed is sampled from here
    #         "lr": lambda: random.uniform(0.0001, 0.02),
    #         'epsilon' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    #         'last_few' : [2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    # })

    # search_space = {
    #     "lr": tune.sample_from(lambda spec:  random.uniform(0.00001, 0.01) ),
    #     'epsilon' : tune.sample_from(lambda s: random.choice([0.0, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])),
    #     "last_few" : tune.sample_from(lambda s: random.choice([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80])),
    #     'epsilon_decay' : tune.sample_from(lambda s: random.choice([i*10_000 for i in range(1, 11)])),
    #     'n_updates_per_iter' : tune.sample_from(lambda s: random.choice([10, 50, 100, 250, 500])),
    #     'max_steps' : 10**7
    # }

    tune.run(
        LunarLanderTrainable,
        # config=search_space,
        # resume=True,
        # Repeat experiments multiple times
        num_samples=100,
        checkpoint_at_end=True #,
        # scheduler=ASHAScheduler(
        #     time_attr='training_iteration',
        #     grace_period=20,
        #     metric="episode_reward_mean", mode="max")
        # TODO: use instead num_episodes_above_threshold_reward as the metric
        #       and threshold_reward=200
    )

    # print("Best config is:", analysis.get_best_config(metric="Buffer_Rewards/mean_last_few"))

if __name__ == "__main__":
    
    main()