from model import MultiInputModel, MultiInputActor, MultiInputCritic
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
            f = self.preprocess(x)
            q = self.model(f)
            a = torch.argmax(q).item()
        return a

    def preprocess(self, x):
        output = {}
        for key in self.state_keys:
            output[key] = torch.from_numpy(x[key]).float().unsqueeze(0).to(self.device)
        return output

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def update_weights(self, checkpoint):
        weights = torch.load(checkpoint)
        model_dict = self.model.state_dict()
        for (k, v), new_v in zip(model_dict.items(), weights.values()):
            model_dict[k] = torch.Tensor(new_v)
        self.model.load_state_dict(model_dict)
        for p in self.model.parameters():
            p.requires_grad = False


class SAC:
    def __init__(self, config):
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
        self.total_steps += 1
        if self.total_steps > self.warmup_steps:
            f = self.preprocess(x)
            a = self.actor(f, deterministic=False, with_logprob=False)[0].detach().cpu().numpy()[0]
        else:
            a = np.random.uniform(-1, 1, self.action_head)
        return a

    def preprocess(self, x):
        output = {}
        for key, value in x.items():
            output[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
        return output

    def update_weights(self, checkpoint):
        # model_dict = self.actor.state_dict()
        # weights =
        # for (k, v), new_v in zip(model_dict.items(), weights.values()):
        #     model_dict[k] = new_v
        self.actor.load_state_dict(torch.load(checkpoint))
        for p in self.actor.parameters():
            p.requires_grad = False

