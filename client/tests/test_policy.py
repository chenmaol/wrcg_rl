
import os
import time
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policy import DQN

base_config = {
    "state":
        {
            "image":
                {
                    "dim": [3, 112, 112],
                    "type": np.uint8,
                    "norm": 255.0,
                },
            "speed":
                {
                    "dim": [1],
                    "type": np.uint8,
                    "norm": 100.0,
                }
        },
    "action":
        {
            "dim": 1,
            "type": np.int64,
            "head": 4
        },
    "reward":
        {
            "dim": 1,
            "type": np.float32,
        },
    "done":
        {
            "dim": 1,
            "type": np.bool_
        }
}

policy = DQN(base_config)

x = {}
x["image"] = np.zeros((3, 112, 112), np.uint8)
x["speed"] = np.zeros((1, ), np.uint8)

# test act
print(policy.act(x, training=True))
print(policy.act(x, training=False))

# test update weights
weights = policy.model.state_dict()

policy.update_weights(weights)
