import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from buffer import ReplayBuffer

with open("../configs/exp1.yaml", 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
base_config = config["base"]
train_config = config["train"]
buffer = ReplayBuffer(base_config, train_config["buffer_size"])

data = {
    "state": {
        "image": np.ones((4, 112, 112), dtype=np.uint8) * 128,
        # "speed": np.ones((1, ), dtype=np.uint8) * 30,
    },
    "action": 1,
    "reward": 5,
    "done": True
}
t0 = time.time()
for i in range(1000):
    buffer.update(data)
t1 = time.time()
print((t1 - t0) / 1000)

for i in range(50):
    sample_data = buffer.sample(256)
t2 = time.time()
print((t2 - t1) / 50)

