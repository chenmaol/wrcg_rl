import time
from model import MultiInputModel, MultiInputCritic, MultiInputActor
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

            next_q = self.target_network(sample_data["state_prime"]).detach()
            current_q = self.network(sample_data["state_prime"]).detach()
            td_target = sample_data["reward"] + \
                        (1. - sample_data["done"]) * self.gamma * next_q.gather(1, torch.max(current_q, 1)[1].unsqueeze(1))
            loss = F.mse_loss(self.network(sample_data["state"]).gather(1, sample_data["action"].long()), td_target)
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

        if total_episodes % self.update_interval == 0:
            while self.update_flag:
                time.sleep(0.1)
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


class SAC:
    def __init__(
            self,
            config,
    ):
        self.config = config

        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        # base config
        self.actor = MultiInputActor(config["model"]).to(self.device)
        self.critic = MultiInputCritic(config["model"]).to(self.device)
        self.target_critic = MultiInputCritic(config["model"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.finetune_ent = True
        if self.finetune_ent:
            self.target_entropy = float(-np.prod(config["action_head"]).astype(np.float32))
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * 1.0).requires_grad_(True)
        else:
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * 0.2)

        # training config
        train_config = self.config["training"]
        self.name = train_config["name"]
        self.lr = train_config["lr"]
        self.gamma = train_config["gamma"]
        self.batch_size = train_config["batch_size"]
        self.warmup_steps = train_config["warmup_steps"]
        self.save_interval = train_config["save_interval"]
        self.update_interval = train_config["update_interval"]
        self.gradient_steps = train_config["gradient_steps"]
        self.reward_deque_length = train_config["reward_deque_length"]
        self.tau = train_config["tau"]

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], self.lr)

        self.total_steps = 0
        self.total_episodes = 0
        self.writer = SummaryWriter(log_dir='./logs/' + self.name)
        self.episode_reward = deque(maxlen=self.reward_deque_length)
        self.episode_len = deque(maxlen=self.reward_deque_length)
        self.update_flag = False

    def learn(self, buffer):
        ent_coef_losses = 0
        ent_coefs = 0
        actor_losses, critic_losses = 0, 0
        # log_probs = []
        # min_qf_pis = []
        # target_q_valueses = []
        # logprob_losses = []
        gradient_steps = self.gradient_steps
        for i in range(gradient_steps):
            sample_data = buffer.sample(self.batch_size)

            actions_pi, log_prob = self.actor(sample_data["state"])
            log_prob = log_prob.reshape(-1, 1)
            # log_probs.append(log_prob.mean().item())

            # log_ent
            if self.finetune_ent:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses += ent_coef_loss.item()

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            else:
                ent_coef = torch.exp(self.log_ent_coef.detach())

            ent_coefs += ent_coef.item()

            # critic
            with torch.no_grad():
                next_actions, next_log_prob = self.actor(sample_data["state_prime"])
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.target_critic(sample_data["state_prime"], next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = sample_data["reward"] + (1 - sample_data["done"]) * self.gamma * next_q_values
                # target_q_valueses.append(target_q_values.mean().item())
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(sample_data["state"], sample_data["action"])

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses += critic_loss.item()
            assert isinstance(critic_loss, torch.Tensor)  # for type checker

            # Optimize the critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = torch.cat(self.critic(sample_data["state"], actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses += actor_loss.item()
            # logprob_losses.append((ent_coef * log_prob).mean().item())
            # min_qf_pis.append(min_qf_pi.mean().item())

            # Optimize the actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.soft_update()
        return {"actor_loss": actor_losses / gradient_steps,
                "critic_loss": critic_losses / gradient_steps,
                "ent_coef_loss": ent_coef_losses / gradient_steps,
                "ent_coef": ent_coefs / gradient_steps,
                }

    def update_network(self, buffer):
        total_steps = self.total_steps
        total_episodes = self.total_episodes

        if total_steps < self.warmup_steps:
            return

        if total_episodes % self.update_interval == 0:
            while self.update_flag:
                time.sleep(0.1)
            self.update_flag = True
            loss = self.learn(buffer)
            self.update_flag = False
            for key, value in loss.items():
                self.writer.add_scalar(key, value, total_steps)

        if total_episodes % self.save_interval == 0:
            torch.save(self.actor.state_dict(), 'checkpoints/sac_actor_{}.pt'.format(total_episodes))

    def update_plot(self, r_seq):
        # update steps
        self.total_steps += len(r_seq)
        self.total_episodes += 1

        # update performance
        self.episode_reward.append(np.sum(r_seq))
        self.episode_len.append(len(r_seq))
        self.writer.add_scalar("rollout/ep_rew_mean", np.mean(list(self.episode_reward)), self.total_steps)
        self.writer.add_scalar("rollout/ep_len_mean", np.mean(list(self.episode_len)), self.total_steps)

    def update(self, buffer, r_seq):
        # update performance for plotting
        self.update_plot(r_seq)
        # update network
        self.update_network(buffer)

    def soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
