from experiment import ex

from sacred.observers import FileStorageObserver


def study():
    ex.observers.append(FileStorageObserver('experiments'))

    for _ in range(3):
        for replay_size in [100, 500]:
            for lr in [0.001, 0.01, 0.1]:
                for n_episodes_per_iter in [100, 10, 1]:
                    for n_updates_per_iter in [200, 500, 1000]:
                        ex.run(config_updates={
                            'eval_every_n_steps': 10_000,                            
                            "env_name" : "LunarLander-v2",
                            "dh" : 200,
                            "dr" : 400,
                            "last_few" : 10,
                            "lr" : lr,
                            "replay_size": replay_size,
                            "n_episodes_per_iter" : n_episodes_per_iter,
                            "n_updates_per_iter" : n_updates_per_iter,
                            "epsilon" : 0.0
                        })

if __name__ == '__main__':
    study()