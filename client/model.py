import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
import numpy as np
import torchvision

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class resnet(torch.nn.Module):
    """ResNet model class for feature extraction."""
    def __init__(self, layers='18', pretrained=False):
        super(resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        """Forward pass through the ResNet model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class conv_bn_relu(torch.nn.Module):
    """Convolutional layer followed by BatchNorm and ReLU activation."""
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size,
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        """Forward pass through the conv_bn_relu block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LaneNet(torch.nn.Module):
    """LaneNet model for lane detection."""
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(LaneNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4
        self.model = resnet(backbone, pretrained=pretrained)

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)

    def forward(self, x):
        """Forward pass through the LaneNet model."""
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2,x3,fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(-1, 1800)

        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls


def initialize_weights(*models):
    """Initialize weights for the given models."""
    for model in models:
        real_init_weights(model)

def real_init_weights(m):
    """Real weight initialization for different layer types."""
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)


class MultiInputMLP(nn.Module):
    """Multi-input MLP for processing various inputs."""
    def __init__(self, config):
        super(MultiInputMLP, self).__init__()
        self.config = config
        input_channel = 201 * 2 * 18  # [img_num_t, 2, 18]
        action_head = config["action_head"]

        self.hid_channel = 128
        self.linear = nn.Sequential(
            nn.Linear(input_channel, self.hid_channel),
            nn.ReLU(),
        )

        if self.config["with_speed"]:
            self.hid_channel += 1
        self.advantage = nn.Linear(self.hid_channel, action_head)
        self.value = nn.Linear(self.hid_channel, 1)

    def forward(self, x):
        """Forward pass through the MultiInputMLP."""
        f = self.linear(x["image"])
        if self.config["with_speed"]:
            f = torch.cat((f, x["speed"]), dim=1)
        advantage = self.advantage(f)
        value = self.value(f)
        return value + advantage - advantage.mean()


class MultiInputModel(nn.Module):
    """Multi-input model for processing image and speed inputs."""
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
        """Forward pass through the MultiInputModel."""
        f = self.cnn(x["image"])
        f = f.view((-1, self.in_features))
        f = self.linear(f)
        if "speed" in self.config["state"]:
            f = torch.cat((f, x["speed"] / 100.), dim=1)
        advantage = self.advantage(f)
        value = self.value(f)
        return value + advantage - advantage.mean()


class MultiInputActor(nn.Module):
    """Multi-input actor model for policy generation."""
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
        """Forward pass through the MultiInputActor."""
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
    """Multi-input critic model for value estimation."""
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
        """Forward pass through the MultiInputCritic."""
        f = self.cnn(x["image"] / self.norm["image"])
        f = f.view((-1, self.in_features))
        f = self.linear(f)
        if self.config["with_speed"]:
            f = torch.cat((f, x["speed"] / self.norm["speed"]), dim=1)
        f = torch.cat([f, a], dim=1)
        return self.q1(f), self.q2(f)
