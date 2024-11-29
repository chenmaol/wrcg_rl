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
import datetime

from policy import DQN, SAC
from buffer import ReplayBuffer


class Server:
    def __init__(self, config_file, run_type="train"):
        # read and save yaml config
        with open(config_file, 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.config["run_type"] = run_type
        with open("config.yaml", 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

        # init policy, buffer
        self.policy = eval(self.config["policy"]["name"])(self.config["policy"])
        self.buffer = {}
        self.policy.buffer = self.buffer

        # if run_type == "train":
            # self.buffer = eval(self.config["buffer"]["name"])(self.config["buffer"])
            # self.policy.buffer = self.buffer
        if run_type == "infer":
            self.policy.load_checkpoint("checkpoints/" + self.config["policy"]["inference"]["checkpoint"])
            self.config["policy"]["training"]["warmup_steps"] = 0

        # init sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 9999))
        sock.listen()

        # create data folder
        if not os.path.exists("data_pool"):
            os.mkdir("data_pool")

        # train thread
        t = Thread(target=self.train)
        t.start()

        self.conn = {}
        self.pbar = {"count": tqdm(desc=f"learning count:{self.policy.count}")}

        # connect thread
        while True:
            conn, addr = sock.accept()
            t = Thread(target=self.new_connection, args=(addr, ))
            t.start()

    def new_connection(self, addr):
        client_ip = addr[0]

        # make new folder
        if not os.path.exists(os.path.join("data_pool", client_ip)):
            os.mkdir(os.path.join("data_pool", client_ip))

        # record connection
        if client_ip not in self.conn:
            self.conn[client_ip] = len(os.listdir(os.path.join("data_pool", client_ip)))
            # create new buffer
            self.buffer[client_ip] = eval(self.config["buffer"]["name"])(self.config["buffer"])
            # create new pbar
            self.pbar[client_ip] = tqdm(total=int(self.config["buffer"]["buffer_size"]), desc=client_ip)

    def train(self):
        while True:
            time.sleep(10)

            # update learning count
            self.pbar["count"].set_description(f"learning count:{self.policy.count}")

            # traverse conn folder, detect new data
            for client_ip, buffer_idx in self.conn.items():
                # put data into buffer
                buffer_num = len(os.listdir(os.path.join("data_pool", client_ip))) - 1
                # print(buffer_num, buffer_idx)
                if buffer_num > buffer_idx:
                    with open(os.path.join("data_pool", client_ip, f"buffer_{buffer_idx}.pkl"), 'rb') as f:
                        data_seq = pickle.load(f)
                        self.buffer[client_ip].update(data_seq)
                        self.policy.update(data_seq["reward"])
                        self.pbar[client_ip].update(len(data_seq["reward"]))

                        if self.pbar[client_ip].n >= self.pbar[client_ip].total:
                            self.pbar[client_ip].n = self.pbar[client_ip].n % self.pbar[client_ip].total
                            self.pbar[client_ip].refresh()

                        self.pbar[client_ip].set_description(f"{client_ip}: {self.buffer[client_ip].data_num}")
                    self.conn[client_ip] = buffer_idx + 1


if __name__ == '__main__':
    server = Server("configs/exp3_sac_mc.yaml")