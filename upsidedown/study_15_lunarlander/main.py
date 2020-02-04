from lunar_lander_trainable import LunarLanderTrainable
import ray
from ray import tune
# from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers import ASHAScheduler
import random
import numpy as np

def main():
    ray.init(num_cpus=4)
   
    tune.run(
        LunarLanderTrainable,
        checkpoint_freq=5,
        checkpoint_at_end=True
    )

    # print("Best config is:", analysis.get_best_config(metric="Buffer_Rewards/mean_last_few"))

if __name__ == "__main__":
    
    main()