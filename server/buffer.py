import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, config, max_size=int(1e5)):
        self.config = config
        self.ptr = 0
        self.size = 0
        self.max_size = int(max_size)
        self.norm = {}
        config["state_prime"] = config["state"]
        self.key_words = ["state", "image", "speed", "reward", "done", "action", "state_prime"]
        self.buffer = self.create_dict_recursively(config)

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
                if "norm" in value:
                    self.norm[key] = value["norm"]
                else:
                    self.norm[key] = 1.0
            else:
                new_dict[key] = self.create_dict_recursively(value)
        return new_dict

    def update_dict_recursively(self, data, buffer):
        for key, value in data.items():
            if isinstance(value, dict):
                self.update_dict_recursively(value, buffer[key])
            else:
                buffer[key][self.ptr] = value

    def sample_dict_recursively(self, buffer, ind):
        output = {}
        for key, value in buffer.items():
            if isinstance(value, dict):
                output[key] = self.sample_dict_recursively(value, ind)
            else:
                output[key] = torch.FloatTensor(value[ind] / self.norm[key])
        return output

    def update(self, data):
        self.update_dict_recursively(data, self.buffer)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        output = self.sample_dict_recursively(self.buffer, ind)
        return output
