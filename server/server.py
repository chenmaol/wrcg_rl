import time
import socket
from tqdm import tqdm
from threading import Thread
import traceback
import numpy as np
import pickle
import yaml
import struct
import os


from policy import DQN, SAC
from buffer import ReplayBuffer


class Server:
    def __init__(self, config_file, run_type="train"):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.config["run_type"] = run_type
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("", 9999))
        self.sock.listen()

        self.run_type = run_type

        self.exp_name = self.config["exp"]["name"]
        self.policy = eval(self.config["policy"]["name"])(self.config["policy"])

        self.policy.load_checkpoint("checkpoints/" + self.config["policy"]["inference"]["checkpoint"])

        if run_type == "train":
            self.buffer = eval(self.config["buffer"]["name"])(self.config["buffer"])
            self.policy.buffer = self.buffer
        if run_type == "infer":
            self.policy.load_checkpoint("checkpoints/" + self.config["policy"]["inference"]["checkpoint"])
            self.config["policy"]["training"]["warmup_steps"] = 0

        self.data_idx = 0

    def start(self):
        with tqdm(ncols=100, leave=True) as pbar:
            pbar.set_description(f"training. remained learn times: {self.policy.count}")
            while True:
                conn, addr = self.sock.accept()

                t = Thread(target=eval("self.{}".format(self.run_type)), args=(conn, addr, pbar))
                t.start()

    def load_pretrain_buffer(self, data_root):
        for data_seq_name in os.listdir(data_root):
            print(data_seq_name)
            with open(os.path.join(data_root, data_seq_name), 'rb') as f:
                data_seq = pickle.load(f)

            self.buffer.update(data_seq)
            r_seq = data_seq["reward"]
            self.policy.update(r_seq)

    def sync_paras(self, conn):
        d = self.policy.actor.state_dict()
        data = {"checkpoint": {}, "total_steps": self.policy.total_steps}
        for k, v in d.items():
            data["checkpoint"][k] = v.cpu().numpy()
        self.send_data(conn, data)

    def init_client(self, conn):
        self.send_data(conn, self.config)

    def train(self, conn, addr, pbar):
        print(f"connected: {addr}")
        try:
            # init start flag and weights for client
            self.init_client(conn)
            self.sync_paras(conn)
            while conn:
                # receive data from client
                data_seq = self.get_data(conn)
                if len(data_seq) == 0:
                    break
                # if len(data_seq["reward"]) <= self.config["env"]["repeat_thres"]:
                #     self.sync_paras(conn)
                #     continue
                # put data into buffer
                self.buffer.update(data_seq)
                r_seq = data_seq["reward"]
                self.policy.update(r_seq)
                # send back start flag and weights
                self.sync_paras(conn)
                # save data into local file
                self.save_data(data_seq)

                pbar.update(len(r_seq))
                pbar.set_description(f"Current Value: {self.policy.count:.2f}")

        except Exception as err:
            print(err)
            print(traceback.print_exc())
        print(f"disconnected: {addr}")

    def infer(self, conn, addr, pbar):
        print(f"connected: {addr}")
        try:
            # init start flag and weights for client
            self.init_client(conn)
            self.sync_paras(conn)
            episode_rewards = []
            while conn:
                # receive data from client
                data = self.get_data(conn)
                if len(data) == 0:
                    break
                self.sync_paras(conn)
                # episode_rewards.append(data["reward"])
                #
                # if data["done"]:
                #     pbar.update(1)
                #     print(np.mean(episode_rewards))
                #     episode_rewards = []

        except Exception as err:
            print(err)
            print(traceback.print_exc())
        print(f"disconnected: {addr}")

    # ================= SOCKET FUNCTION ==================
    def get_data(self, conn):
        conn.settimeout(self.config["policy"]["wait_time"])
        data_len = struct.unpack('>Q', conn.recv(8))[0]
        data = b''
        while len(data) < data_len:
            packet = conn.recv(min(data_len - len(data), 4096000))
            if not packet:
                break
            data += packet

        if len(data) == data_len:
            return pickle.loads(data)
        else:
            raise Exception('数据接收不完整')

    def send_data(self, conn, data):
        data = pickle.dumps(data)
        data_len = len(data)

        conn.sendall(struct.pack('>Q', data_len))
        conn.sendall(data)

    def save_data(self, data):
        data_idx = self.data_idx
        self.data_idx += 1
        with open(f"../../data_pool/data_pool/{self.exp_name}_{data_idx}.pkl", 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    server = Server("configs/exp1_sac_newzealand_v1.9.0.yaml", 'train')
    # server.load_pretrain_buffer("processed_data")
    server.start()
