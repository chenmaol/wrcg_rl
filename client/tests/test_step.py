import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import Env
env = Env()

while True:
    t0 = time.time()
    action = np.random.randint(0, 4)
    s, r, done, end = env.step(action)
    t1 = time.time()
    print(t1 - t0)
    if end:
        env.reset_game()
    if done:
        env.reset_car()

