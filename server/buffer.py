import os.path

import numpy as np
import torch
import pickle
import threading


class ReplayBuffer:
    """Class to manage a replay buffer for storing and sampling experiences."""
    
    def __init__(self, config):
        """Initialize the replay buffer with configuration settings."""
        self.config = config
        self.ptr = 0
        self.size = 0
        self.max_size = int(config["buffer_size"])

        self.save_interval = self.max_size // 100
        self.data_num = 0
        self.save_ptr = 0

        config["state_prime"] = config["state"]

        self.key_words = ["state", "image", "speed", "reward", "done", "action", "state_prime"]
        self.buffer = self.create_dict_recursively(config)

        self.exp_name = config["exp_name"]

        if not os.path.exists("buffer"):
            os.mkdir("buffer")

    def create_dict_recursively(self, d):
        """Create a dictionary recursively based on the configuration."""
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
        """Update the buffer dictionary recursively with new data sequences."""
        for key, value in data_seq.items():
            if isinstance(value, dict):
                self.update_dict_recursively(value, buffer[key])
            else:
                update_len = min(self.max_size - self.ptr, len(value))
                buffer[key][self.ptr:self.ptr + update_len] = value[:update_len]

    def sample_dict_recursively(self, buffer, ind):
        """Sample data from the buffer dictionary recursively based on indices."""
        output = {}
        for key, value in buffer.items():
            if isinstance(value, dict):
                output[key] = self.sample_dict_recursively(value, ind)
            else:
                output[key] = torch.FloatTensor(value[ind]).to("cuda")
        return output

    def update(self, data_seq):
        """Update the replay buffer with new data sequences."""
        self.update_dict_recursively(data_seq, self.buffer)

        update_len = min(self.max_size - self.ptr, len(data_seq["reward"]))
        self.ptr = (self.ptr + update_len) % self.max_size
        self.size = min(self.size + update_len, self.max_size)
        self.data_num += update_len
        #
        # while self.data_num >= (self.save_ptr + 1) * self.save_interval:
        #     self.save_buffer()

    def sample(self, batch_size):
        """Sample a batch of experiences from the replay buffer."""
        ind = np.random.randint(0, self.size, batch_size)
        output = self.sample_dict_recursively(self.buffer, ind)
        return output

    # def save_buffer(self):
    #     """Save the current state of the buffer to disk."""
    #     # np.save(f"buffer/{self.exp_name}_{self.buffer_idx}.npy", self.buffer)
    #     threading.Thread(target=self._save_buffer).start()
    #
    # def _save_buffer(self):
    #     """Threaded function to save the buffer to disk."""
    #     # np.save(f"buffer/{self.exp_name}_{self.buffer_idx}.npy", self.buffer)
    #     buffer_idx = self.save_ptr
    #     self.save_ptr += 1
    #
    #     sub_buffer = self.extract_first_n_elements(self.buffer, buffer_idx)
    #     with open(f"buffer/{self.exp_name}_{buffer_idx}.pkl", 'wb') as f:
    #         pickle.dump(sub_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # def extract_first_n_elements(self, data, idx, new_data=None):
    #     """Extract the first n elements from the buffer based on the index."""
    #     if new_data is None:
    #         new_data = {}
    #
    #     if isinstance(data, dict):
    #         for key, value in data.items():
    #             new_data[key] = self.extract_first_n_elements(value, idx, {})
    #     else:
    #         start_idx = (idx * self.save_interval) % self.max_size
    #         end_idx = ((idx + 1) * self.save_interval) % self.max_size
    #         if end_idx == 0:
    #             end_idx = self.max_size
    #         new_data = data[start_idx:end_idx]
    #
    #     return new_data