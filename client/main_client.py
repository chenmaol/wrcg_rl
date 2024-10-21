from env import Env
from agent_client import Agent
from numpysocket import NumpySocket
import numpy as np
import time
import torch

def single_run():
    ret = 0
    s = wrcg_env.reset_car()
    for i in range(max_seqlen):
        a = agent.act(s, training=False)
        s_prime, r, done, end = wrcg_env.step(a)
        if end:
            s = wrcg_env.reset_game()
            break
        if done:
            break
        s = s_prime
        ret += r
    return np.round(ret, 4)

def evaluate(checkpoint_steps, eval_iterations=5):
    agent.network.load_state_dict(torch.load('checkpoints/dqn_{}.pt'.format(checkpoint_steps)))
    for eval_t in range(eval_iterations):
        cum_r = single_run()
        print(cum_r)


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
            if done or end:
                agent.get_data(socket)


if __name__ == '__main__':
    input_channel = 3
    action_dim = 4
    max_seqlen = 1000
    states_with_speed = True

    wrcg_env = Env()
    agent = Agent(input_channel, action_dim)
    # train(wrcg_env, agent, "10.19.226.34", 9999)

    print(evaluate(160000))