import os
import time
import socket
from tqdm import tqdm
from threading import Thread
import traceback
import numpy as np
import pickle
import yaml
import struct


from policy import DQN, SAC, OfflineSAC
from buffer import ReplayBuffer


class OfflineTrainer:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.exp_name = self.config["exp"]["name"]
        self.policy = eval("Offline"+self.config["policy"]["name"])(self.config["policy"])

        # self.policy.load_checkpoint("checkpoints/" + self.config["policy"]["inference"]["checkpoint"])

        self.buffer = eval(self.config["buffer"]["name"])(self.config["buffer"])
        self.policy.buffer = self.buffer

        self.data_idx = 0

    def train(self, data_dirs):
        total_steps = [0 for i in range(len(data_dirs))]
        steps = [0 for i in range(len(data_dirs))]
        indices = [0 for i in range(len(data_dirs))]
        total_data_paths = []
        for i, data_dir in enumerate(data_dirs):
            data_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
            data_paths = sorted(data_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            total_data_paths.append(data_paths.copy())
            for data_path in tqdm(data_paths, total=len(data_paths)):
                with open(data_path, 'rb') as f:
                    data_seq = pickle.load(f)
                total_steps[i] += len(data_seq["reward"])
        for data_dir, cnt in zip(data_dirs, total_steps):
            print(data_dir, cnt)

        for k in range(1, 1000 + 1):
            cur_ratio = k / 1000
            print(f"K:{cur_ratio}")
            for i, data_dir in enumerate(data_dirs):
                while steps[i] / total_steps[i] < cur_ratio:
                    cur_index = indices[i]
                    with open(total_data_paths[i][cur_index], 'rb') as f:
                        data_seq = pickle.load(f)
                        steps[i] += len(data_seq["reward"])
                        indices[i] += 1
                        print(total_data_paths[i][cur_index], steps[i], indices[i])

                        self.buffer.update(data_seq)
                        r_seq = data_seq["reward"]
                        self.policy.update(r_seq)


if __name__ == '__main__':
    data_dir = ["data_pool_v1.8.0_japan", "data_pool_v1.8.0_wales"]
    trainer = OfflineTrainer("configs/exp1_sac_mixed_v1.8.0.yaml")
    trainer.train(data_dir)
