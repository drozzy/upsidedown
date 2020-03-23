from cartpole_trainable import CartPoleTrainable
import ray
from ray import tune
# from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers import ASHAScheduler
import random
import numpy as np

def main():
    ray.init(num_cpus=4)

    tune.run(
        CartPoleTrainable,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        # restore="/home/andriy/ray_results/CartPoleTrainable/CartPoleTrainable_86edf4c0_2020-02-02_20-56-13lpgolfq2/checkpoint_813/checkpoint.pt"
    )

    # print("Best config is:", analysis.get_best_config(metric="Buffer_Rewards/mean_last_few"))

if __name__ == "__main__":
    
    main()