import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import Env

env = Env()

while True:
    image = env.get_frame()
    speed = env.get_speed(image)
    print(speed)

