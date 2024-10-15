import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.uint8)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )

    def save(self):
        print('saving buffer')
        np.save('saved_buffer/s.npy', self.s)
        np.save('saved_buffer/a.npy', self.a)
        np.save('saved_buffer/r.npy', self.r)
        np.save('saved_buffer/s_prime.npy', self.s_prime)
        np.save('saved_buffer/terminated.npy', self.terminated)
        np.save('saved_buffer/para.npy', np.array([self.ptr, self.size, self.max_size]))

    def load(self):
        print('loading buffer')
        self.s = np.load('saved_buffer/s.npy')
        self.a = np.load('saved_buffer/a.npy')
        self.r = np.load('saved_buffer/r.npy')
        self.s_prime = np.load('saved_buffer/s_prime.npy')
        self.terminated = np.load('saved_buffer/terminated.npy')
        self.ptr, self.size, self.max_size = np.load('saved_buffer/para.npy')