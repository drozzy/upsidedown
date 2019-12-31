from experiment import ex

from sacred.observers import FileStorageObserver


def study():
    ex.observers.append(FileStorageObserver('experiments'))

    # Repeat a few times
    hidden_sizes = [8, 32, 128, 8, 32, 128, 8, 32, 128]

    for hidden_size in hidden_sizes:        
            ex.run(config_updates={
                'hidden_size': hidden_size,
                'max_steps' : 100
            })

if __name__ == '__main__':
    study()