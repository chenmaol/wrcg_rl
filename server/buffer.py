import os.path

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.ptr = 0
        self.size = 0
        self.max_size = int(config["buffer_size"])
        config["state_prime"] = config["state"]

        self.key_words = ["state", "image", "speed", "reward", "done", "action", "state_prime"]
        self.buffer = self.create_dict_recursively(config)

        self.exp_name = config["exp_name"]
        self.buffer_idx = 0
        if not os.path.exists("buffer"):
            os.mkdir("buffer")

    def create_dict_recursively(self, d):
        new_dict = {}
        for key, value in d.items():
            if key not in self.key_words:
                continue
            if "dim" in value:
                dims = [self.max_size]
                dims += value["dim"] if isinstance(value["dim"], list) else [value["dim"]]
                dtype = eval(value["type"])
                new_dict[key] = np.zeros(dims, dtype=dtype)

            else:
                new_dict[key] = self.create_dict_recursively(value)
        return new_dict

    def update_dict_recursively(self, data_seq, buffer):
        for key, value in data_seq.items():
            if isinstance(value, dict):
                self.update_dict_recursively(value, buffer[key])
            else:
                update_len = min(self.max_size - self.ptr, len(value))
                buffer[key][self.ptr:self.ptr + update_len] = value[:update_len]

    def sample_dict_recursively(self, buffer, ind):
        output = {}
        for key, value in buffer.items():
            if isinstance(value, dict):
                output[key] = self.sample_dict_recursively(value, ind)
            else:
                output[key] = torch.FloatTensor(value[ind]).to("cuda")
        return output

    def update(self, data_seq):
        self.update_dict_recursively(data_seq, self.buffer)

        previous_ptr = self.ptr
        update_len = min(self.max_size - self.ptr, len(data_seq["reward"]))
        self.ptr = (self.ptr + update_len) % self.max_size
        self.size = min(self.size + update_len, self.max_size)

        if self.ptr < previous_ptr:
            self.save_buffer()

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        output = self.sample_dict_recursively(self.buffer, ind)
        return output

    def save_buffer(self):
        np.save(f"buffer/{self.exp_name}_{self.buffer_idx}.npy", self.buffer)
        self.buffer_idx += 1
