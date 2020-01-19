from lunar_lander_trainable import LunarLanderTrainable
import ray
from ray import tune

def main():
    ray.init()
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
        time_attr='time_total_s',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=600.0,
        hyperparam_mutations={
            "lr": lambda: random.uniform(0.0001, 0.02),
            'epsilon' : lambda: random.uniform(0.0, 1.0),
            'last_few' : [2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    })
    
    tune.run(
        LunarLanderTrainable,
        config={
            'max_steps' : 10**3
        },
        # Repeat experiments multiple times
        num_samples=5,
        checkpoint_at_end=True,
        max_failures=0,
        scheduler=pbt_scheduler)

    # print("Best config is:", analysis.get_best_config(metric="Buffer_Rewards/mean_last_few"))

if __name__ == "__main__":
    
    main()