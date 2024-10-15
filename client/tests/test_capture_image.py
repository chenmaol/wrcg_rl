import cv2

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import Env


env = Env()
image = env.get_frame()
print(image.shape)
cv2.imwrite("test_capture_image.jpg", image)