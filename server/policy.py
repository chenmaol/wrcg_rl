import time
from model import MultiInputModel, MultiInputCritic, MultiInputActor, MultiInputMLP
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import threading
import logging

logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class BasePolicy:
    def __init__(self, config):
        ''' Initialize the base policy with configuration settings '''
        self.config = config
        self.update_count = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.writer = SummaryWriter(log_dir='./logs/' + config["training"]["name"])
        self.episode_reward = deque(maxlen=config["training"]["reward_deque_length"])
        self.episode_len = deque(maxlen=config["training"]["reward_deque_length"])
        self.warmup_steps = config["training"]["warmup_steps"]
        self.update_interval = config["training"]["update_interval"]
        self.save_interval = config["training"]["save_interval"]

    def update_plot(self, r_seq):
        ''' Update the plot with the given reward sequence '''
        # update steps
        self.total_steps += len(r_seq)
        self.total_episodes += 1

        # update performance
        self.episode_reward.append(np.sum(r_seq))
        self.episode_len.append(len(r_seq))
        self.writer.add_scalar("rollout/ep_rew_mean", np.mean(list(self.episode_reward)), self.total_steps)
        self.writer.add_scalar("rollout/ep_len_mean", np.mean(list(self.episode_len)), self.total_steps)

    def load_checkpoint(self, checkpoint):
        ''' Load the model checkpoint from the specified file '''
        if "DQN" in self.config["name"]:
            self.network.load_state_dict(torch.load(checkpoint))
        elif "SAC" in self.config["name"]:
            self.actor.load_state_dict(torch.load(checkpoint))

    def save_checkpoint(self, checkpoint_path):
        ''' Save the model checkpoint to the specified file path '''
        if "DQN" in self.config["name"]:
            torch.save(self.network.state_dict(), checkpoint_path)
        elif "SAC" in self.config["name"]:
            torch.save(self.actor.state_dict(), checkpoint_path)

    def get_checkpoint(self):
        ''' Retrieve the current model checkpoint '''
        if "DQN" in self.config["name"]:
            return self.network.state_dict()
        elif "SAC" in self.config["name"]:
            return self.actor.state_dict()

    def learn_thread(self):
        ''' Thread function for learning process '''
        while True:
            while self.update_count < 1:
                time.sleep(1)
            if self.update_count > 0:
                self.update_count -= 1
                loss = self.learn()
                for key, value in loss.items():
                    self.writer.add_scalar(key, value, self.total_steps)

    def update_network(self, seq_len, total_steps, total_episodes):
        ''' Update the network based on the sequence length and total steps '''
        if total_steps >= self.warmup_steps:
            self.update_count += seq_len / self.update_interval
        
        if total_episodes % self.save_interval == 0:
            checkpoint_path = 'checkpoints/{}_{}.pt'.format(self.config["policy"]["name"], total_episodes)
            self.save_checkpoint(checkpoint_path)

    def update(self, r_seq):
        ''' Update the policy with the given reward sequence '''
        total_steps = self.total_steps
        total_episodes = self.total_episodes

        # update performance for plotting
        self.update_plot(r_seq)
        # update network
        self.update_network(len(r_seq), total_steps, total_episodes)

    def sync(self):
        ''' Synchronize the current state of the policy '''
        d = self.get_checkpoint()
        data = {"checkpoint": {}, "total_steps": self.total_steps}
        for k, v in d.items():
            data["checkpoint"][k] = v.cpu().numpy()
        return data

class DQN(BasePolicy):
    def __init__(
            self,
            config,
    ):
        ''' Initialize the DQN policy with configuration settings '''
        super().__init__(config)  # 确保调用基类构造函数
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        # base config
        self.network = MultiInputMLP(config["model"]).to(self.device)
        self.target_network = MultiInputMLP(config["model"]).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        # training config
        train_config = self.config["training"]
        self.name = train_config["name"]
        self.lr = train_config["lr"]
        self.epsilon = train_config["epsilon"]
        self.epsilon_min = train_config["epsilon_min"]
        self.epsilon_steps = train_config["epsilon_steps"]
        self.gamma = train_config["gamma"]
        self.batch_size = train_config["batch_size"]
        self.target_update_interval = train_config["target_update_interval"]
        self.update_interval = train_config["update_interval"]
        self.gradient_steps = train_config["gradient_steps"]
        self.reward_deque_length = train_config["reward_deque_length"]

        self.optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_steps

        self.thread = threading.Thread(target=self.learn_thread)
        self.thread.daemon = True  # 设置为守护线程，以便主程序退出时子线程也会退出
        self.thread.start()

    def learn(self):
        ''' Perform a learning step and return the loss '''
        mean_loss = 0
        for i in range(self.gradient_steps):
            sample_data = self.buffer.sample(self.batch_size)

            next_q = self.target_network(sample_data["state_prime"]).detach()
            current_q = self.network(sample_data["state_prime"]).detach()
            td_target = sample_data["reward"] + \
                        (1. - sample_data["done"]) * self.gamma * next_q.gather(1,
                                                                                torch.max(current_q, 1)[1].unsqueeze(1))
            loss = F.mse_loss(self.network(sample_data["state"]).gather(1, sample_data["action"].long()), td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss += loss.item()
        return {"loss": mean_loss / self.gradient_steps}

    def update_network(self, seq_len, total_steps, total_episodes):
        ''' Update the DQN network based on the sequence length and total steps '''
        self.update_count += seq_len / self.update_interval
        if total_steps < self.warmup_steps:
            return

        if total_episodes % self.save_interval == 0:
            checkpoint_path = 'checkpoints/{}_{}.pt'.format(self.config["name"], total_episodes)
            self.save_checkpoint(checkpoint_path)

        # DQN
        if total_episodes % self.target_update_interval:
            self.hard_update()

        self.update_epsilon(seq_len)

    def update_plot(self, r_seq):
        ''' Update the plot with the given reward sequence and epsilon value '''
        super().update_plot(r_seq)
        self.writer.add_scalar("para/epsilon", self.epsilon, self.total_steps)

    def update_epsilon(self, episode_len):
        ''' Update the epsilon value based on the episode length '''
        self.epsilon = max(self.epsilon - self.epsilon_decay * episode_len, self.epsilon_min)

    def hard_update(self):
        ''' Perform a hard update of the target network '''
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(param.data)

    def sync(self):
        ''' Synchronize the current state of the DQN policy '''
        data = super().sync()
        data["epsilon"] = self.epsilon
        return data

class SAC(BasePolicy):
    def __init__(
            self,
            config,
    ):
        ''' Initialize the SAC policy with configuration settings '''
        super().__init__(config)  # 确保调用基类构造函数
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

        self.learn_thread = threading.Thread(target=self.learn_thread)
        self.learn_thread.daemon = True  # 设置为守护线程，以便主程序退出时子线程也会退出
        self.learn_thread.start()

    def learn(self):
        ''' Perform a learning step and return the losses for actor, critic, and entropy coefficient '''
        ent_coef_losses = 0
        ent_coefs = 0
        actor_losses, critic_losses = 0, 0
        # log_probs = []
        # min_qf_pis = []
        # target_q_valueses = []
        # logprob_losses = []
        gradient_steps = self.gradient_steps
        for i in range(gradient_steps):
            sample_data = self.buffer.sample(self.batch_size)

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

    def soft_update(self):
        ''' Perform a soft update of the target critic network '''
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


class OfflineSAC(SAC):
    def __init__(self, config):
        ''' Initialize the Offline SAC policy with configuration settings '''
        super().__init__(config)
        self.offline_count = 0

    def update_network(self, seq_len):
        ''' Update the network for offline learning based on the sequence length '''
        total_steps = self.total_steps
        total_episodes = self.total_episodes

        if total_steps < self.warmup_steps:
            return

        self.offline_count += seq_len / self.update_interval

        while self.offline_count >= 1:
            loss = self.learn()
            for key, value in loss.items():
                self.writer.add_scalar(key, value, self.total_steps)
            self.offline_count -= 1
        if total_episodes % self.save_interval == 0:
            checkpoint = 'checkpoints/sac_actor_{}.pt'.format(total_episodes)
            torch.save(self.actor.state_dict(), checkpoint)
