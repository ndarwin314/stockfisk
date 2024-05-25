import asyncio
import random
import torch.multiprocessing as mp
import torch
import numpy as np
from minimax_q.worker import Learner, Actor, ReplayBuffer
from minimax_q.minimaxq import Minimax
from minimax_q.r2d2 import Network
import minimax_q.config as config


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)
mp.set_start_method('fork')


def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (num_actors-1) * alpha
    return base_eps**exponent


def wrap_run(actor):
    asyncio.run(actor.run())


def train(num_actors=config.num_actors, log_interval=config.log_interval):
    env = Minimax(None, None)

    model = Network(env.action_space.shape[0], env.observation_space.shape[0], config.hidden_dim)
    del env
    model.share_memory()
    sample_queue_list = [mp.Queue() for _ in range(num_actors)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(sample_queue_list, batch_queue, priority_queue)
    learner = Learner(batch_queue, priority_queue, model)
    actors = [Actor(model, get_epsilon(i), sample_queue_list[i]) for i in range(num_actors)]

    actor_procs = [mp.Process(target=wrap_run, args=(actor,)) for actor in actors]
    for proc in actor_procs:
        proc.start()

    buffer_proc = mp.Process(target=buffer.run)
    buffer_proc.start()

    learner.run()

    buffer_proc.join()

    for proc in actor_procs:
        proc.terminate()



