from lunar_lander import ex

from sacred.observers import FileStorageObserver


def study():
    ex.observers.append(FileStorageObserver('studies'))

    epsilons = [0.0, 0.01, 0.05]
    max_returns = [1, 10, 100, 300]

    for epsilon in epsilons:
        for max_return in max_returns:
            ex.run(config_updates={
                'start_epsilon': epsilon,
                'max_return' : max_return
            })

if __name__ == '__main__':
    study()