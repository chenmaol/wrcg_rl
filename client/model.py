import torch.nn as nn
import torch.nn.functional as F
import torch


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
