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
                "last_few" : 40,
                "lr" : 0.001,
                "replay_size": 40,
                "n_episodes_per_iter" : 40,
                "n_updates_per_iter" : 100,
                "epsilon" : 0.0
            })

if __name__ == '__main__':
    study()