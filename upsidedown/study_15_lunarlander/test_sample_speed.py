import time
from lib import ReplayBuffer, Trajectory
import torch
def main():
	rb = ReplayBuffer(max_size=100, last_few=10)
	tj = []
	for i in range(100):
		t = Trajectory()
		for j in range(10):
			t.add(prev_action=j, state=j, action=j, reward=j, state_prime=j)			
		tj.append(t)

	rb.add(tj)

	batch_size = 2048
	num_samples = 250
	device = torch.device('cuda')
	start = time.time()

	for i in range(num_samples):
		rb.sample(batch_size, device)
	end = time.time()

	print(f'Took {end - start} to num_samples={num_samples}, batch_size={batch_size}')

if __name__ == '__main__':
	main()