import torch.nn as nn
import torch.nn.functional as F
class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CNNActionValue, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(state_dim, 16, kernel_size=4, stride=2),  # [N, 4, 112, 112] -> [N, 16, 55, 55] PF=4
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
        self.advantage = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view((-1, self.in_features))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()