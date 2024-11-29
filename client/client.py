import struct
import pickle
import copy
import os
import yaml
import socket
from scp import SCPClient
import paramiko

from policy import DQN, SAC
from env import WRCGDiscreteEnv, WRCGContinuousEnv


# ################### SSH Function ######################
def get_local_ip(target_ip, target_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((target_ip, target_port))
    IP = s.getsockname()[0]
    return IP


def create_ssh_client(server_ip, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server_ip, username=username, password=password)
    return ssh


def download_files(ssh, local_download_path, remote_download_path):
    with SCPClient(ssh.get_transport()) as scp:
        # Download a file
        scp.get(remote_download_path, local_download_path)
        print(f"Downloaded {remote_download_path} to {local_download_path}")


def upload_files(ssh, local_upload_path, remote_upload_path):
    with SCPClient(ssh.get_transport()) as scp:
        # Upload a file
        scp.put(local_upload_path, remote_upload_path)
        print(f"Uploaded {local_upload_path} to {remote_upload_path}")


# ################### Client ######################
class Client:
    def __init__(self):
        self.root = root
        self.ip = get_local_ip(ip, port)
        print(self.ip)
        self.buffer_idx = 0

        # download config.yaml from server -> load config file
        self.download_config()
        with open("config.yaml", 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        # init policy, env, buffer
        self.policy = eval(self.config['policy']["name"])(self.config["policy"])
        self.env = eval(self.config["env"]["name"])(self.config["env"])
        self.buffer = self.init_data_buffer()

        # run type
        self.env.run_type = self.config["run_type"]
        if self.config["run_type"] == "infer":
            self.env.repeat_thres = self.config['policy']['inference']["repeat_thres"]

        # sync weights
        self.sync_weights()

    # ################### Client upload / download func ######################
    def download_config(self):
        download_files(ssh, "config.yaml", os.path.join(self.root, "config.yaml"))

    def download_weights(self):
        download_files(ssh, "actor.pt", os.path.join(self.root, "actor.pt"))

    def upload_data(self):
        upload_files(ssh, "buffer.pkl", os.path.join(self.root, "data_pool", self.ip, f"buffer_{self.buffer_idx}.pkl"))

    def sync_weights(self):
        self.download_weights()
        self.policy.update_weights("actor.pt")

    def save_buffer(self):
        with open("buffer.pkl", 'wb') as f:
            pickle.dump(self.buffer, f)
        self.clear_data_buffer()
        self.upload_data()

    # ################### buffer func ######################
    def init_data_buffer(self):
        buffer = {}
        for key, value in self.config["buffer"].items():
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
        for key, value in data.items():
            if key in self.buffer and isinstance(self.buffer[key], list):
                self.buffer[key].append(value)
            elif key in self.buffer and isinstance(self.buffer[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key in self.buffer[key] and isinstance(self.buffer[key][sub_key], list):
                        self.buffer[key][sub_key].append(sub_value)

    def clear_data_buffer(self):
        for key, value in self.buffer.items():
            if key in self.buffer and isinstance(self.buffer[key], list):
                self.buffer[key].clear()
            elif key in self.buffer and isinstance(self.buffer[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key in self.buffer[key] and isinstance(self.buffer[key][sub_key], list):
                        self.buffer[key][sub_key].clear()

    # ################### train func ######################
    def train(self):
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
                    self.save_buffer()
                    self.sync_weights()
                    break

    def eval(self, checkpoint_path):
        pass


if __name__ == '__main__':
    ip = "10.78.0.78"
    port = 9999
    username = "dlfg"
    password = "Atp67$23h100new"
    root = "/localhome/dlfg/chenmao/wrcg_rl/server/"
    # init ssh
    ssh = create_ssh_client(ip, username, password)

    # init client
    client = Client()

    # train
    print("start training")
    client.train()
