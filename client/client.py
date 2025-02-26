import socket
import struct
import pickle
import copy
from policy import DQN, SAC
from env import WRCGDiscreteEnv, WRCGContinuousEnv, WRCGLaneEnv

import logging

logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Client:
    """Client class to handle socket communication and training with the environment."""
    
    def __init__(self, ip, port):
        """Initialize the Client with IP and port, set up socket connection and configuration."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))

        self.wait_time = 60.0
        self.config = self.get_data()
        self.policy = eval(self.config['policy']["name"])(self.config["policy"])
        self.env = eval(self.config["env"]["name"])(self.config["env"])
        # run type
        self.env.run_type = self.config["run_type"]
        if self.config["run_type"] == "infer":
            self.env.repeat_thres = self.config['policy']['inference']["repeat_thres"]

        self.wait_time = self.config['policy']["wait_time"]

        self.sync_paras()

        self.buffer = self.init_data_buffer(self.config["buffer"])

    def init_data_buffer(self, buffer_config):
        """Initialize the data buffer based on the provided configuration."""
        buffer = {}
        for key, value in buffer_config.items():
            if key not in ["state", "action", "reward", "done"]:
                continue
            if key != "state":
                buffer[key] = []
            else:
                buffer[key] = {}
                for sub_key, sub_value in value.items():
                    buffer[key][sub_key] = []
        buffer["state_prime"] = copy.deepcopy(buffer["state"])
        return buffer

    def update_data_buffer(self, data):
        """Update the data buffer with new data received from the environment."""
        for key, value in data.items():
            if key in self.buffer and isinstance(self.buffer[key], list):
                self.buffer[key].append(value)
            elif key in self.buffer and isinstance(self.buffer[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key in self.buffer[key] and isinstance(self.buffer[key][sub_key], list):
                        self.buffer[key][sub_key].append(sub_value)

    def clear_data_buffer(self):
        """Clear the data buffer to reset its state."""
        for key, value in self.buffer.items():
            if key in self.buffer and isinstance(self.buffer[key], list):
                self.buffer[key].clear()
            elif key in self.buffer and isinstance(self.buffer[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key in self.buffer[key] and isinstance(self.buffer[key][sub_key], list):
                        self.buffer[key][sub_key].clear()

    def sync_paras(self):
        """Synchronize parameters with the received data from the server."""
        received_data = self.get_data()
        self.policy.sync(received_data)

    def train(self):
        """Main training loop for the client, interacting with the environment."""
        while True:
            # reset car
            s = self.env.reset_car()
            for i in range(self.config["policy"]["training"]["max_episode_length"]):
                # select action by state
                a = self.policy.act(s)
                # get next states, reward, done, by env
                data = self.env.step(a)
                data["state"] = s.copy()

                # update each step
                s = data["state_prime"]

                # save data to local buffer
                self.update_data_buffer(data)
                if data["done"]:
                    self.send_data()
                    logging.info(f"sent data")
                    self.sync_paras()
                    logging.info(f"got data")
                    break

    def eval(self, checkpoint_path):
        """Evaluate the model using the specified checkpoint path."""
        pass

    # ================= SOCKET FUNCTION ==================
    def get_data(self):
        """Receive data from the server and return it after unpacking."""
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

    def send_data(self):
        """Send the current data buffer to the server."""
        data = pickle.dumps(self.buffer)
        data_len = len(data)

        self.sock.sendall(struct.pack('>Q', data_len))
        self.sock.sendall(data)

        self.clear_data_buffer()


if __name__ == '__main__':
    ip, port = "10.19.226.34", 9999

    # init
    client = Client(ip, port)

    # train
    print("start training")
    client.train()