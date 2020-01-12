from experiment import ex

from sacred.observers import FileStorageObserver


def study():
    ex.observers.append(FileStorageObserver('experiments'))

    repeat = 10
    
    for _ in range(repeat):
        ex.run(config_updates={
                "env_name" : "LunarLander-v2",
                'eval_every_n_steps': 10_000,                            
                "dh" : 400,
                "dr" : 400,
                "last_few" : 50,
                "lr" : 0.005,
                "replay_size": 100,
                "n_episodes_per_iter" : 500,
                "n_updates_per_iter" : 200,
                "epsilon" : 0.0,
                "return_scale" : 0.01,
                "horizon_scale" : 0.001
        
            })

if __name__ == '__main__':
    study()