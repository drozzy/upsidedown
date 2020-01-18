from experiment import ex

from sacred.observers import FileStorageObserver


def study():
    ex.observers.append(FileStorageObserver('experiments'))

    repeat = 10
    
    for return_scale in [0.01, 0.001, 0.1]:
        for horizon_scale in [0.001, 0.01, 0.1]:
            for num_stack in [10, 5, 2, 50]:
                for lr in [0.001, 0.005, 0.01]:
                    for n_updates_per_iter in [50, 100]:
                        for last_few in [10, 20]:
                            for _ in range(repeat):

                                replay_size = last_few*5 
                                n_episodes_per_iter = last_few*5

                                experiment_name = f'lr_{lr}_nu_{n_updates_per_iter}_lf_{last_few}_rs_{replay_size}_ne_{n_episodes_per_iter}_rscale_{return_scale}_hscale_{horizon_scale}_nstack_{num_stack}'
                                ex.run(config_updates={
                                        "experiment_name" : experiment_name,
                                        "env_name" : "LunarLander-v2",
                                        'eval_every_n_steps': 10_000,                            
                                        "dh" : 400,
                                        "dr" : 400,
                                        "num_stack" : num_stack,
                                        "last_few" : last_few,
                                        "lr" : lr,
                                        "replay_size": replay_size,
                                        "n_episodes_per_iter" : n_episodes_per_iter,
                                        "n_updates_per_iter" : n_updates_per_iter,
                                        "epsilon" : 0.0,
                                        "max_steps" : 10**6,
                                        "return_scale" : return_scale,
                                        "horizon_scale" : horizon_scale
                                    })

if __name__ == '__main__':
    study()