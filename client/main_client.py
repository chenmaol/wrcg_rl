from env import Env
from agent_client import Agent
from numpysocket import NumpySocket
import numpy as np
import time
import torch

def evaluate(wrcg_env, agent, steps):
    ret = 0
    agent.network.load_state_dict(torch.load('checkpoints/dqn_{}.pt'.format(steps)))
    s = wrcg_env.reset_game()
    for i in range(1000):
        a = agent.act(s, training=False)
        s_prime, r, done, err = wrcg_env.step(a)
        if done or err == 1 or err == 2:
            s = wrcg_env.reset_car()
            continue
        if err == 3:
            s = wrcg_env.reset_game()
            continue
        s = s_prime
        ret += r
    return np.round(ret, 4)


def train(wrcg_env, agent, ip, port):
    with NumpySocket() as socket:
        socket.connect((ip, port))
        agent.get_data(socket)
        print('update weights')
        while True:
            # reset car
            s = wrcg_env.reset_car()
            for i in range(max_seqlen):
                # select action by state
                a = agent.act(s)
                # get next states, reward, done, by env
                s_prime, r, done, end = wrcg_env.step(a)
                # send data to server
                agent.send_data((s, a, r, s_prime, done or end), socket)
                # update each step
                s = s_prime
                if end:
                    s = wrcg_env.reset_game()
                    break
                if done:
                    break
            agent.get_data(socket)


if __name__ == '__main__':
    input_channel = 4
    action_dim = 4
    max_seqlen = 1000

    wrcg_env = Env()
    agent = Agent(input_channel, action_dim)
    train(wrcg_env, agent, "10.19.226.34", 9999)

    # print(evaluate(wrcg_env, agent, 160000))