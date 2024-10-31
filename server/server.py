import time
import socket
from tqdm import tqdm
from threading import Thread
import traceback
import numpy as np
import pickle
import yaml
import struct

from policy import DQN, SAC
from buffer import ReplayBuffer


class Server:
    def __init__(self, config_file, run_type="train"):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 9999))
        sock.listen()

        self.policy = eval(self.config["base"]["policy"])(self.config)
        self.buffer = ReplayBuffer(self.config["base"], self.config["train"]["buffer_size"])

        with tqdm(ncols=100, leave=True) as pbar:
            pbar.set_description("training")
            while True:
                conn, addr = sock.accept()

                t = Thread(target=eval("self.{}".format(run_type)), args=(conn, addr, pbar))
                t.start()

    def sync_paras(self, conn):
        d = self.policy.network.state_dict()
        data = {"checkpoint": {}, "epsilon": self.policy.epsilon}
        for k, v in d.items():
            data["checkpoint"][k] = v.cpu().numpy()
        self.send_data(conn, data)

    def init_client(self, conn):
        self.send_data(conn, self.config["base"])

    def train(self, conn, addr, pbar):
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
                # put data into buffer
                self.buffer.update(data)
                episode_rewards.append(data["reward"])

                if data["done"]:
                    self.policy.update(self.buffer, episode_rewards)
                    # send back start flag and weights
                    self.sync_paras(conn)
                    episode_rewards = []
                pbar.update(1)

        except Exception as err:
            print(err)
            print(traceback.print_exc())
        print(f"disconnected: {addr}")

    # ================= SOCKET FUNCTION ==================
    def get_data(self, conn):
        conn.settimeout(self.config["base"]["wait_time"])
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


if __name__ == '__main__':
    server = Server("configs/exp1.yaml")
