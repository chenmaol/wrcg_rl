import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import socket
from client import Client

ip, port = "10.19.226.34", 9999

# init
client = Client(ip, port)

# train
client.train()

# eval
client.eval()
