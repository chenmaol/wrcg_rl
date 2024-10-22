import time
from model import MultiInputModel
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class DQN:
    def __init__(
            self,
            config,
    ):
        base_config = config["base"]
        train_config = config["train"]
        self.base_config = base_config

        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        # base config
        self.network = MultiInputModel(base_config).to(self.device)
        self.target_network = MultiInputModel(base_config).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # training config
        self.name = train_config["name"]
        self.lr = train_config["lr"]
        self.epsilon = train_config["epsilon"]
        self.epsilon_min = train_config["epsilon_min"]
        self.epsilon_steps = train_config["epsilon_steps"]
        self.gamma = train_config["gamma"]
        self.batch_size = train_config["batch_size"]
        self.warmup_steps = train_config["warmup_steps"]
        self.target_update_interval = train_config["target_update_interval"]
        self.update_interval = train_config["update_interval"]
        self.gradient_steps = train_config["gradient_steps"]
        self.reward_deque_length = train_config["reward_deque_length"]

        self.optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        self.total_steps = 0
        self.total_episodes = 0
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_steps
        self.writer = SummaryWriter(log_dir='./logs/' + self.name)
        self.episode_reward = deque(maxlen=self.reward_deque_length)
        self.episode_len = deque(maxlen=self.reward_deque_length)
        self.update_flag = False

    def learn(self, buffer):
        mean_loss = 0
        K = self.gradient_steps
        for i in range(K):
            sample_data = buffer.sample(self.batch_size)

            next_q = self.target_network(sample_data["state_prime"]).to(self.device).detach()
            current_q = self.network(sample_data["state_prime"]).to(self.device).detach()
            td_target = sample_data["reward"].to(self.device) + \
                        (1. - sample_data["done"].to(self.device)) * self.gamma * next_q.gather(1, torch.max(current_q, 1)[1].unsqueeze(1))
            loss = F.mse_loss(self.network(sample_data["state"].to(self.device)).gather(1, sample_data["action"].to(self.device).long()), td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss += loss.item()
        return mean_loss / K

    def update_network(self, buffer):
        total_steps = self.total_steps
        total_episodes = self.total_episodes

        if total_steps < self.warmup_steps:
            return

        while self.update_flag:
            time.sleep(0.1)

        if total_episodes % self.update_interval == 0:
            self.update_flag = True
            loss = self.learn(buffer)
            self.update_flag = False
            self.writer.add_scalar('loss', loss, total_steps)

        if total_episodes % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            torch.save(self.network.state_dict(), 'checkpoints/dqn_{}.pt'.format(total_steps))

    def update_plot(self, r_seq):
        # update steps
        self.total_steps += len(r_seq)
        self.total_episodes += 1

        # update performance
        self.episode_reward.append(np.sum(r_seq))
        self.episode_len.append(len(r_seq))
        self.writer.add_scalar("rollout/ep_rew_mean", np.mean(list(self.episode_reward)), self.total_steps)
        self.writer.add_scalar("rollout/ep_len_mean", np.mean(list(self.episode_len)), self.total_steps)
        self.writer.add_scalar("rollout/exploration_rate", self.epsilon, self.total_steps)

    def update_epsilon(self, episode_len):
        self.epsilon = max(self.epsilon - self.epsilon_decay * episode_len, self.epsilon_min)

    def update(self, buffer, r_seq):
        # update performance for plotting
        self.update_epsilon(len(r_seq))
        # update performance for plotting
        self.update_plot(r_seq)
        # update network
        self.update_network(buffer)
