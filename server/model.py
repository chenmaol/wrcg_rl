# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from torch.distributions.normal import Normal
#
# LOG_STD_MAX = 2
# LOG_STD_MIN = -20
#
#
# class MultiInputModel(nn.Module):
#     def __init__(self, config):
#         super(MultiInputModel, self).__init__()
#         self.config = config
#         input_channel = config["state"]["image"]["dim"][0]
#         action_head = config["action"]["head"]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channel, 16, kernel_size=4, stride=2),  # [N, 4, 112, 112] -> [N, 16, 55, 55] PF=4
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2),  # [N, 16, 55, 55] -> [N, 32, 27, 27] PF=8
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2),  # [N, 32, 27, 27] -> [N, 64, 13, 13] PF=16
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2),  # [N, 64, 13, 13] -> [N, 64, 6, 6] PF=32
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),  # [N, 64, 6, 6] -> [N, 64, 2, 2] PF=64
#             nn.ReLU(),
#         )
#
#         self.in_features = 256 * 2 * 2
#         self.hid_channel = 64
#         self.linear = nn.Sequential(
#             nn.Linear(self.in_features, self.hid_channel),
#             nn.ReLU(),
#         )
#
#         if "speed" in self.config["state"]:
#             self.hid_channel += 1
#         self.advantage = nn.Linear(self.hid_channel, action_head)
#         self.value = nn.Linear(self.hid_channel, 1)
#
#     def forward(self, x):
#         f = self.cnn(x["image"])
#         f = f.view((-1, self.in_features))
#         f = self.linear(f)
#         if "speed" in self.config["state"]:
#             f = torch.cat((f, x["speed"]), dim=1)
#         advantage = self.advantage(f)
#         value = self.value(f)
#         return value + advantage - advantage.mean()
#
#
# class MultiInputActor(nn.Module):
#     def __init__(self, config):
#         super(MultiInputActor, self).__init__()
#         self.config = config
#         input_channel = config["input_channel"]
#         action_head = config["action_head"]
#         self.norm = config["norm"]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channel, 16, kernel_size=4, stride=2),  # [N, 4, 112, 112] -> [N, 16, 55, 55] PF=4
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2),  # [N, 16, 55, 55] -> [N, 32, 27, 27] PF=8
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2),  # [N, 32, 27, 27] -> [N, 64, 13, 13] PF=16
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2),  # [N, 64, 13, 13] -> [N, 64, 6, 6] PF=32
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),  # [N, 64, 6, 6] -> [N, 64, 2, 2] PF=64
#             nn.ReLU(),
#         )
#
#         self.in_features = 256 * 5 * 5
#         self.hid_channel = 128
#
#         self.speed_embed = nn.Sequential(
#             nn.Linear(1, self.hid_channel),
#             nn.ReLU(),
#         )
#         self.image_fc = nn.Sequential(
#             nn.Linear(self.in_features, self.hid_channel),
#             nn.ReLU(),
#         )
#
#         self.mu = nn.Linear(self.hid_channel * 2, action_head)
#         self.log_std = nn.Linear(self.hid_channel * 2, action_head)
#
#         self.epsilon = 1e-6
#
#     def forward(self, x, deterministic=False, with_logprob=True):
#         f = self.cnn(x["image"] / self.norm["image"])
#         f = f.view((-1, self.in_features))
#         f = self.image_fc(f)
#         if self.config["with_speed"]:
#             sf = self.speed_embed(x["speed"] / self.norm["speed"])
#             f = torch.cat((f, sf), dim=1)
#
#         mu = self.mu(f)
#         log_std = self.log_std(f)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)
#
#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             # Only used for evaluating policy at test time.
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()
#
#         action = torch.tanh(pi_action)
#         if with_logprob:
#             # logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             # logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
#             # log_prob = torch.sum(pi_distribution.log_prob(pi_action), dim=1)
#             # log_prob -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + self.epsilon), dim=1)
#             log_prob = torch.sum(pi_distribution.log_prob(pi_action), dim=1)
#             log_prob -= torch.sum(torch.log(1 - action ** 2 + self.epsilon), dim=1)
#         else:
#             log_prob = None
#
#         return action, log_prob
#
#
# class MultiInputCritic(nn.Module):
#     def __init__(self, config):
#         super(MultiInputCritic, self).__init__()
#         self.config = config
#         input_channel = config["input_channel"]
#         action_head = config["action_head"]
#         self.norm = config["norm"]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channel, 16, kernel_size=4, stride=2),  # [N, 4, 224, 224] -> [N, 16, 55, 55] PF=4
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2),  # [N, 16, 55, 55] -> [N, 32, 27, 27] PF=8
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2),  # [N, 32, 27, 27] -> [N, 64, 13, 13] PF=16
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2),  # [N, 64, 13, 13] -> [N, 64, 6, 6] PF=32
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),  # [N, 64, 6, 6] -> [N, 64, 2, 2] PF=64
#             nn.ReLU(),
#         )
#
#         self.in_features = 256 * 5 * 5
#         self.hid_channel = 128
#
#         self.speed_embed = nn.Sequential(
#             nn.Linear(1, self.hid_channel),
#             nn.ReLU(),
#         )
#         self.image_fc = nn.Sequential(
#             nn.Linear(self.in_features, self.hid_channel),
#             nn.ReLU(),
#         )
#
#         self.q1 = nn.Sequential(
#             nn.Linear(self.hid_channel * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )
#         self.q2 = nn.Sequential(
#             nn.Linear(self.hid_channel * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )
#
#     def forward(self, x, a):
#         f = self.cnn(x["image"] / self.norm["image"])
#         f = f.view((-1, self.in_features))
#         f = self.image_fc(f)
#         if self.config["with_speed"]:
#             sf = self.speed_embed(x["speed"] / self.norm["speed"])
#             f = torch.cat((f, sf), dim=1)
#         return self.q1(f), self.q2(f)

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MultiInputModel(nn.Module):
    def __init__(self, config):
        super(MultiInputModel, self).__init__()
        self.config = config
        input_channel = config["state"]["image"]["dim"][0]
        action_head = config["action"]["head"]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=4, stride=2),  # [N, 4, 112, 112] -> [N, 16, 55, 55] PF=4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # [N, 16, 55, 55] -> [N, 32, 27, 27] PF=8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # [N, 32, 27, 27] -> [N, 64, 13, 13] PF=16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # [N, 64, 13, 13] -> [N, 64, 6, 6] PF=32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # [N, 64, 6, 6] -> [N, 64, 2, 2] PF=64
            nn.ReLU(),
        )

        self.in_features = 256 * 2 * 2
        self.hid_channel = 64
        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.hid_channel),
            nn.ReLU(),
        )

        if "speed" in self.config["state"]:
            self.hid_channel += 1
        self.advantage = nn.Linear(self.hid_channel, action_head)
        self.value = nn.Linear(self.hid_channel, 1)

    def forward(self, x):
        f = self.cnn(x["image"])
        f = f.view((-1, self.in_features))
        f = self.linear(f)
        if "speed" in self.config["state"]:
            f = torch.cat((f, x["speed"]), dim=1)
        advantage = self.advantage(f)
        value = self.value(f)
        return value + advantage - advantage.mean()


class MultiInputActor(nn.Module):
    def __init__(self, config):
        super(MultiInputActor, self).__init__()
        self.config = config
        input_channel = config["input_channel"]
        action_head = config["action_head"]
        self.norm = config["norm"]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=4, stride=2),  # [N, 4, 112, 112] -> [N, 16, 55, 55] PF=4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # [N, 16, 55, 55] -> [N, 32, 27, 27] PF=8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # [N, 32, 27, 27] -> [N, 64, 13, 13] PF=16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # [N, 64, 13, 13] -> [N, 64, 6, 6] PF=32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # [N, 64, 6, 6] -> [N, 64, 2, 2] PF=64
            nn.ReLU(),
        )

        self.in_features = 256 * 5 * 5
        self.hid_channel = 64
        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.hid_channel),
            nn.ReLU(),
        )

        if self.config["with_speed"]:
            self.hid_channel += 1

        self.mu = nn.Linear(self.hid_channel, action_head)
        self.log_std = nn.Linear(self.hid_channel, action_head)

        self.epsilon = 1e-6

    def forward(self, x, deterministic=False, with_logprob=True):
        f = self.cnn(x["image"] / self.norm["image"])
        f = f.view((-1, self.in_features))
        f = self.linear(f)
        if self.config["with_speed"]:
            f = torch.cat((f, x["speed"] / self.norm["speed"]), dim=1)

        mu = self.mu(f)
        log_std = self.log_std(f)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        action = torch.tanh(pi_action)
        if with_logprob:
            # logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
            # log_prob = torch.sum(pi_distribution.log_prob(pi_action), dim=1)
            # log_prob -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + self.epsilon), dim=1)
            log_prob = torch.sum(pi_distribution.log_prob(pi_action), dim=1)
            log_prob -= torch.sum(torch.log(1 - action ** 2 + self.epsilon), dim=1)
        else:
            log_prob = None

        return action, log_prob


class MultiInputCritic(nn.Module):
    def __init__(self, config):
        super(MultiInputCritic, self).__init__()
        self.config = config
        input_channel = config["input_channel"]
        action_head = config["action_head"]
        self.norm = config["norm"]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=4, stride=2),  # [N, 4, 112, 112] -> [N, 16, 55, 55] PF=4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # [N, 16, 55, 55] -> [N, 32, 27, 27] PF=8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # [N, 32, 27, 27] -> [N, 64, 13, 13] PF=16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # [N, 64, 13, 13] -> [N, 64, 6, 6] PF=32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # [N, 64, 6, 6] -> [N, 64, 2, 2] PF=64
            nn.ReLU(),
        )

        self.in_features = 256 * 5 * 5
        self.hid_channel = 64
        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.hid_channel),
            nn.ReLU(),
        )

        if self.config["with_speed"]:
            self.hid_channel += 1
        self.q1 = nn.Sequential(
            nn.Linear(self.hid_channel + action_head, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.hid_channel + action_head, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, a):
        f = self.cnn(x["image"] / self.norm["image"])
        f = f.view((-1, self.in_features))
        f = self.linear(f)
        if self.config["with_speed"]:
            f = torch.cat((f, x["speed"] / self.norm["speed"]), dim=1)
        f = torch.cat([f, a], dim=1)
        return self.q1(f), self.q2(f)
