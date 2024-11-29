import cv2

import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import WRCGContinuousEnv

with open("../config.yaml", 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

env = WRCGContinuousEnv(config["env"])
image = env.get_frame()
print(image.shape)
cv2.imwrite("test_capture_image.jpg", image)