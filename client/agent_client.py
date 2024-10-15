from model import CNNActionValue
import numpy as np
import torch

class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
    ):
        self.action_dim = action_dim
        self.network = CNNActionValue(state_dim, action_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.epsilon = 1
        self.wait_time = 60.0

    @torch.no_grad()
    def act(self, x, training=True):
        if training and np.random.rand() < self.epsilon:
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a

    def send_data(self, transition, socket):
        s, a, r, s_prime, done = transition
        socket.settimeout(self.wait_time)
        mixed = np.array([s, a, r, s_prime, done], dtype=object)
        socket.sendall(mixed)

    def get_data(self, socket):
        socket.settimeout(self.wait_time)
        new_values = socket.recv()
        self.epsilon = new_values[-1]
        model_dict = self.network.state_dict()
        for (k, v), new_v in zip(model_dict.items(), new_values[:-1]):
            model_dict[k] = torch.Tensor(new_v)
        self.network.load_state_dict(model_dict)
        for p in self.network.parameters():
            p.requires_grad = False



