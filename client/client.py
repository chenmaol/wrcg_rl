import socket
import struct
import pickle

from policy import DQN, SAC
from env import WRCGDiscreteEnv, WRCGContinuousEnv


class Client:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))

        self.wait_time = 60.0

        self.base_config = self.get_data()
        self.policy = eval(self.base_config['policy'])(self.base_config)
        self.env = eval(self.base_config['env'])(self.base_config)

        self.sync_paras()

    def sync_paras(self):
        received_data = self.get_data()
        self.policy.update_weights(received_data["checkpoint"])

    def train(self):
        while True:
            # reset car
            s = self.env.reset_car()
            for i in range(self.base_config["max_episode_length"]):
                # select action by state
                a = self.policy.act(s)
                # get next states, reward, done, by env
                data = self.env.step(a)
                data["state"] = s
                data["action"] = a
                # send data to server
                self.send_data(data)
                # update each step
                s = data["state_prime"]
                if data["done"]:
                    self.sync_paras()
                    break

    def eval(self, checkpoint_path):
        pass

    # ================= SOCKET FUNCTION ==================
    def get_data(self):
        self.sock.settimeout(self.wait_time)
        data_len = struct.unpack('>Q', self.sock.recv(8))[0]
        data = b''
        while len(data) < data_len:
            packet = self.sock.recv(min(data_len - len(data), 4096000))
            if not packet:
                break
            data += packet

        if len(data) == data_len:
            return pickle.loads(data)
        else:
            raise Exception('数据接收不完整')

    def send_data(self, data):
        data = pickle.dumps(data)
        data_len = len(data)

        self.sock.sendall(struct.pack('>Q', data_len))
        self.sock.sendall(data)


if __name__ == '__main__':
    ip, port = "10.19.226.34", 9999

    # init
    client = Client(ip, port)

    # train
    print("start training")
    client.train()