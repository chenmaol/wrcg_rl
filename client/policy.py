from model import MultiInputModel
import numpy as np
import torch


class DQN:
    def __init__(self, config):
        self.device = "cuda"
        self.epsilon = 1.0

        self.model = MultiInputModel(config).to(self.device)
        self.action_head = config["action"]["head"]
        self.state_keys = config["state"].keys()
        self.config = config


    @torch.no_grad()
    def act(self, x, training=True):
        if training and np.random.rand() < self.epsilon:
            a = np.random.randint(0, self.action_head)
        else:
            x = self.preprocess(x)
            q = self.model(x)
            a = torch.argmax(q).item()
        return a

    def preprocess(self, x):
        for key in self.state_keys:
            x[key] = torch.from_numpy(x[key]).float().unsqueeze(0).to(self.device) / self.config["state"][key]["norm"]
        return x

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def update_weights(self, weights):
        model_dict = self.model.state_dict()
        for (k, v), new_v in zip(model_dict.items(), weights.values()):
            model_dict[k] = torch.Tensor(new_v)
        self.model.load_state_dict(model_dict)
        for p in self.model.parameters():
            p.requires_grad = False




