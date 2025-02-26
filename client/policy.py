from model import MultiInputMLP, MultiInputActor, MultiInputCritic
import numpy as np
import torch


class DQN:
    """DQN class for Deep Q-Learning algorithm."""
    def __init__(self, config):
        """Initialize the DQN with configuration settings."""
        self.device = "cuda"

        self.epsilon = 0.8
        self.total_steps = 0

        self.model = MultiInputMLP(config["model"]).to(self.device)
        self.warmup_steps = config["training"]["warmup_steps"]
        self.action_head = config["action_head"]
        self.config = config

    @torch.no_grad()
    def act(self, x, training=True):
        """Select an action based on the current state and training mode."""
        if training and (self.total_steps < self.warmup_steps or np.random.rand() < self.epsilon):
            a = np.random.randint(0, self.action_head)
        else:
            x = self.preprocess(x)
            q = self.model(x)
            a = torch.argmax(q).item()
        return a

    def preprocess(self, x):
        """Preprocess the input data for the model."""
        output = {}
        for key, value in x.items():
            output[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
        return output

    def update_epsilon(self, epsilon):
        """Update the exploration rate (epsilon)."""
        self.epsilon = epsilon

    def update_weights(self, weights):
        """Update the model weights from the provided weights."""
        model_dict = self.model.state_dict()
        for (k, v), new_v in zip(model_dict.items(), weights.values()):
            model_dict[k] = torch.Tensor(new_v)
        self.model.load_state_dict(model_dict)
        for p in self.model.parameters():
            p.requires_grad = False

    def sync(self, data):
        """Synchronize the model with the provided checkpoint data."""
        self.update_weights(data["checkpoint"])
        self.total_steps = data["total_steps"]
        self.update_epsilon(data["epsilon"])

class SAC:
    """SAC class for Soft Actor-Critic algorithm."""
    def __init__(self, config):
        """Initialize the SAC with configuration settings."""
        self.device = "cuda"
        self.warmup_steps = config["training"]["warmup_steps"]
        self.total_steps = 0

        self.actor = MultiInputActor(config["model"]).to(self.device)

        for p in self.actor.parameters():
            p.requires_grad = False

        self.action_head = config["action_head"]

        self.config = config

    @torch.no_grad()
    def act(self, x, training=True):
        """Select an action based on the current state and training mode."""
        self.total_steps += 1
        if self.total_steps > self.warmup_steps:
            f = self.preprocess(x)
            a = self.actor(f, deterministic=False, with_logprob=False)[0].detach().cpu().numpy()[0]
        else:
            a = np.random.uniform(-1, 1, self.action_head)
        return a

    def preprocess(self, x):
        """Preprocess the input data for the actor."""
        output = {}
        for key, value in x.items():
            output[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
        return output

    def update_weights(self, weights):
        """Update the actor weights from the provided weights."""
        model_dict = self.actor.state_dict()
        for (k, v), new_v in zip(model_dict.items(), weights.values()):
            model_dict[k] = torch.Tensor(new_v)
        self.actor.load_state_dict(model_dict)
        for p in self.actor.parameters():
            p.requires_grad = False

