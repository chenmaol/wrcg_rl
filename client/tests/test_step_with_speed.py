import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import Env
from agent_client import Agent
input_channel = 3
action_dim = 4
max_seqlen = 1000
states_with_speed = True
agent = Agent(input_channel, action_dim, states_with_speed=states_with_speed)
env = Env(states_with_speed=True, num_concat_image=1, gray_scale=False)

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

