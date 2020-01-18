from lunar_lander_trainable import LunarLanderTrainable
import ray
from ray import tune

def main():
    ray.init()
    analysis = tune.run(
        LunarLanderTrainable,
        config={
            'max_steps' : 10**3
        },
        # Repeat experiments multiple times
        num_samples=10,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        max_failures=1
    )

    print("Best config is:", analysis.get_best_config(metric="Buffer_Rewards/mean_last_few"))

if __name__ == "__main__":
    
    main()