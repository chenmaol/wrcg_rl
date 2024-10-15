import time
from numpysocket import NumpySocket
from agent_server import Agent
from tqdm import tqdm
from threading import Thread
import traceback


def process(conn, addr, pbar):
    print(f"connected: {addr}")
    try:
        # init start flag and weights for client
        agent.send_data(conn)
        episode_rewards = []
        while conn:
            # receive data from client
            data = agent.get_data(conn)
            if len(data) == 0:
                break
            # put data into buffer
            s, a, r, s_prime, terminated = data
            agent.buffer.update(s, a, r, s_prime, terminated)
            episode_rewards.append(r)

            if terminated:
                # update performance for plotting
                agent.update_plot(episode_rewards)
                # update network
                agent.update_network(len(episode_rewards))
                # send back start flag and weights
                agent.send_data(conn)
                episode_rewards = []
            pbar.update(1)

    except Exception as err:
        print(err)
        print(traceback.print_exc())
    print(f"disconnected: {addr}")


if __name__ == '__main__':
    stack_frames = 4
    action_dim = 4
    img_size = 112

    state_dim = (stack_frames, img_size, img_size)
    agent = Agent(state_dim, action_dim)

    continue_steps = 0

    max_steps = 2e5

    with NumpySocket() as socket:
        socket.bind(("", 9999))
        socket.listen()
        with tqdm(total=int(max_steps), ncols=100, leave=True) as pbar:
            pbar.set_description("training")
            pbar.update(continue_steps)
            while True:
                conn, addr = socket.accept()
                t = Thread(target=process, args=(conn, addr, pbar))
                t.start()

