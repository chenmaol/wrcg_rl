import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import Env
from agent_client import Agent

input_channel = 4
action_dim = 4
env = Env()
agent = Agent(input_channel, action_dim)
s = env.reset_car()
while True:
    t0 = time.time()
    a = agent.act(s, training=False)
    s, r, done, end = env.step(a)
    t1 = time.time()
    print(t1 - t0)
    if end:
        env.reset_game()
    if done:
        env.reset_car()

