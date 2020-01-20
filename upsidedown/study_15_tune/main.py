from lunar_lander_trainable import LunarLanderTrainable
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import random

def main():
    ray.init(num_cpus=3)
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

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=5, # 600.0,
        hyperparam_mutations={
            # Prob of this mutation being performed is sampled from here
            "lr": lambda: random.uniform(0.0001, 0.02),
            'epsilon' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
            'last_few' : [2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    })

    tune.run(
        LunarLanderTrainable,
        config={
            "gpu": 0.25,
            "last_few" : tune.grid_search([10, 50]),
            'lr': tune.grid_search([0.001]),
            'epsilon' : tune.grid_search([0.0, 0.9]),
            'max_steps' : 10**7
        },
        # resume=True,
        # Repeat experiments multiple times
        # num_samples=5,
        checkpoint_at_end=True,
        max_failures=0,
        verbose=False,
        scheduler=pbt_scheduler)

    # print("Best config is:", analysis.get_best_config(metric="Buffer_Rewards/mean_last_few"))

if __name__ == "__main__":
    
    main()