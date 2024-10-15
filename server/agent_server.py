import time

from model import CNNActionValue
import torch.nn.functional as F
import torch
from buffer import ReplayBuffer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.00025,
            epsilon=0.8,
            epsilon_min=0.1,
            gamma=0.99,
            batch_size=256,
            warmup_steps=5000,
            buffer_size=int(5e4),
            update_interval=5,
            target_update_interval=100,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.update_interval = update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.total_episodes = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e5
        self.epsilon_min = epsilon_min
        self.wait_time = 60.0
        self.writer = SummaryWriter(log_dir='./logs/exp')
        self.episode_reward = deque(maxlen=1000)
        self.episode_len = deque(maxlen=1000)
        self.update_flag = False

    def learn(self):
        mean_loss = 0
        K = 50
        for i in range(K):
            s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

            s /= 255.
            s_prime /= 255.

            next_q = self.target_network(s_prime).detach()
            current_q = self.network(s_prime).detach()
            td_target = r + (1. - terminated) * self.gamma * next_q.gather(1, torch.max(current_q, 1)[1].unsqueeze(1))
            loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss += loss.item()
        return mean_loss / K

    def update_network(self, episode_len):
        total_steps = self.total_steps
        total_episodes = self.total_episodes
        self.epsilon = max(self.epsilon - self.epsilon_decay * episode_len, self.epsilon_min)

        if total_steps < self.warmup_steps:
            return

        while self.update_flag:
            time.sleep(0.1)

        if total_episodes % self.update_interval == 0:
            self.update_flag = True
            loss = self.learn()
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

    def get_data(self, conn):
        conn.settimeout(self.wait_time)
        s, a, r, s_prime, done = conn.recv()
        return s, a, r, s_prime, done

    def send_data(self, conn):
        d = self.network.state_dict()
        array_list = [x.cpu().numpy() for x in list(d.values())]
        array_list.append(self.epsilon)
        mixed = np.array(array_list, dtype=object)
        conn.sendall(mixed)
