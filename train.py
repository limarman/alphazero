import time
from multiprocessing import Queue, Lock, Event, Value, Process, Manager

from typing import NamedTuple
import numpy as np
import torch
import os

import psutil

from connect4 import Connect4State
from model import SimpleTakeAwayModel, TicTacToeModel, Connect4Model, Connect4Model_64conv, Connect4Model_128conv
from montecarlo import TakeAwayState
from selfplay import self_play, play_tournament, async_self_play
import torch.nn.functional as F

from ticktactoe import TicTacToeState

class DataBufferSamples(NamedTuple):
    states: torch.Tensor
    pis: torch.Tensor
    zs: torch.Tensor
    

class Databuffer:

    states: np.ndarray
    pis: np.ndarray
    zs: np.ndarray

    def __init__(self, buffer_size: int, state_shape, action_shape):
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, ) + state_shape, dtype=np.float32)
        self.pis = np.zeros((buffer_size, ) + action_shape, dtype=np.float32)
        self.zs = np.zeros((buffer_size, 1), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add_element(self, state, pi, z):
        self.states[self.pos] = state
        self.pis[self.pos] = pi
        self.zs[self.pos] = z
        self.pos += 1

        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = self.pos % self.buffer_size

    def sample_batch(self, batch_size: int):
        if self.full:
            batch_inds = np.random.randint(0, self.buffer_size-1, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        return self._sample(batch_inds)

    def _sample(self, indices):

        data = (
            self.states[indices, :],
            self.pis[indices, :],
            self.zs[indices, :]
        )
        return DataBufferSamples(*tuple(map(self.to_torch, data)))
    
    def to_torch(self, array: np.ndarray, copy: bool = True):
        if copy:
            return torch.tensor(array)
        return torch.as_tensor(array)


def save_model(model, name):

    if not os.path.exists("models"):
        os.makedirs("models")

    path = f"models/{name}"

    torch.save(model, path)


def self_play_worker(task_queue, result_queue, ready_event, worker_id, game_start_state, data_queue, n_selfplays):
    while True:
        states, pis, winners = async_self_play(task_queue=task_queue, result_queue=result_queue,
                                               ready_event=ready_event, worker_id=worker_id,
                                               start_state=game_start_state, mcts_simulations=100, temperature_tau=1)

        data_queue.put((states, pis, winners))
        #print(data_queue.qsize())

        if data_queue.qsize() >= n_selfplays:
            #print(f"Terminating worker {worker_id}")
            break

        #with counter.get_lock():
        #    counter.value += 1

def nn_worker(task_queue, result_queues, ready_events, model):
    while True:
        batch = []
        while len(batch) < 8 and not task_queue.empty():
            worker_id, state = task_queue.get()
            batch.append((worker_id, state))

        if len(batch) != 0:
            #print(len(batch))
            with torch.no_grad():
                # create and pass the batch of data
                states = np.stack([model.construct_state_representation(state) for worker_id, state in batch])
                states = torch.tensor(states, requires_grad=False)
                probs_logits, vals = model.forward(states)
                probss = F.softmax(probs_logits, dim=1)

            for (worker_id, _), probs, val in zip(batch, probss, vals):

                prob_dict = {model.map_index_to_actionname(index): probs[index] for index in
                             range(len(probs))}

                result_queues[worker_id].put((worker_id, (prob_dict, val)))
                #time.sleep(2)
                ready_events[worker_id].set()  # Signal that NN processing is done for this worker


if __name__ == '__main__':

    # Settings for resuming training procedure
    new_arch = True
    load_model_path = "Connect4/buffer30k_64conv/generation_5.pt"
    add_gen = 0
    n_restart_selfplays = 300

    n_selfplays = 100
    n_gradient_steps = 5000 #10000
    n_iterations = 1
    n_epochs = 100

    buffer_size = 30000 #10000 #6000 #3000
    batch_size = 128

    game_start_state = Connect4State() #TicTacToeState() #TakeAwayState(20)

    ModelArchitecture = Connect4Model_128conv

    if load_model_path is not None:
        data_generating_model = torch.load(load_model_path)
    else:
        data_generating_model = ModelArchitecture()

    trained_model = ModelArchitecture() #Connect4Model() # TicTacToeModel() # SimpleTakeAwayModel()

    if not new_arch or load_model_path is None:

        # initialize the same
        trained_model.load_state_dict(data_generating_model.state_dict())

    optimizer = torch.optim.Adam(params=trained_model.parameters(), lr=1e-3, weight_decay=1e-4)

    #databuffer = Databuffer(buffer_size=buffer_size, state_shape=(2,3,3), action_shape=(9,))
    databuffer = Databuffer(buffer_size=buffer_size, state_shape=(2,6,7), action_shape=(7,))

    gen = 0 if load_model_path is None else add_gen

    for g in range(n_epochs):

        print(f"Epoch {g+1} / Generation {gen}")

        for i in range(n_iterations):

            if i % 100 == 0:
                print(f"Starting Iteration {i}...")

            """for j in range(n_selfplays):
                states, pis, winners = self_play(game_start_state, data_generating_model, mcts_simulations=100, temperature_tau=1)

                for state, pi, z in zip(states, pis, winners):
                    
                    # generate the full pi for the agent (including moves that are not an option)
                    full_pi = np.zeros(len(game_start_state.action_space))

                    for key,value in pi.items():
                        full_pi[trained_model.map_actionname_to_index(key)] = value

                    databuffer.add_element(trained_model.construct_state_representation(state), full_pi, z)
            """

            # ASYNC VARIANT
            no_self_play_workers = 8
            no_nn_workers = 1

            task_queue = Queue()
            result_queues = [Queue() for _ in range(no_self_play_workers)]
            ready_events = [Event() for _ in range(no_self_play_workers)]

            data_queue = Manager().Queue()
            #buffer_lock = Lock()
            #self_play_counter = Value('i', 0)

            self_play_start = time.time()

            self_play_workers = [Process(target=self_play_worker, args=(
            task_queue, result_queues[j], ready_events[j], j, game_start_state, data_queue, n_restart_selfplays if i == 0 and g == 0 and load_model_path is not None else n_selfplays))
                                 for j in range(no_self_play_workers)]
            for worker in self_play_workers:
                worker.start()

            # Start NN worker
            nn_workers = [Process(target=nn_worker, args=(task_queue, result_queues, ready_events, data_generating_model))
                          for i in range(no_nn_workers)]
            for worker in nn_workers:
                worker.start()
                psutil.Process(worker.pid).nice(psutil.HIGH_PRIORITY_CLASS)  # Set high priority

            """try:
                while True:
                    time.sleep(1)  # Main thread can perform other tasks or monitoring
                    #print(f"Self-plays finished: {data_queue.qsize()}")
                    if data_queue.qsize() >= n_selfplays:
                        break
            except KeyboardInterrupt:
                pass
            finally:
                for worker in self_play_workers:
                    worker.join()
                for worker in nn_workers:
                    worker.terminate()
                for worker in nn_workers:
                    worker.join()"""

            for worker in self_play_workers:
                worker.join()
            for worker in nn_workers:
                worker.terminate()
            for worker in nn_workers:
                worker.join()


            self_play_finish = time.time()
            print(f"Selfplay duration: {self_play_finish - self_play_start}")
            print(f"Selfplay/s: {data_queue.qsize() / (self_play_finish - self_play_start)}")

            #print(data_queue.qsize())

            # Collect data from result queue
            while not data_queue.empty():
                states, pis, winners = data_queue.get()
                data_queue.qsize()
                for state, pi, z in zip(states, pis, winners):
                    # generate the full pi for the agent (including moves that are not an option)
                    full_pi = np.zeros(len(game_start_state.action_space))

                    for key, value in pi.items():
                        full_pi[trained_model.map_actionname_to_index(key)] = value

                    for state, pi in trained_model.get_alike_state_data(state, full_pi):
                        databuffer.add_element(trained_model.construct_state_representation(state), pi, z)

            #print(data_queue.qsize())

            for j in range(n_gradient_steps):

                data = databuffer.sample_batch(batch_size=batch_size)

                pred_pis_logits, pred_vals = trained_model(data.states)

                # construct the loss function
                loss = F.mse_loss(pred_vals, data.zs) + F.cross_entropy(pred_pis_logits, data.pis)

                if j % 100 == 0:
                    print(f'Loss: {loss.item()}')

                if j == n_gradient_steps-1:
                    print(data.states[:10, :])
                    print(data.pis[:10, :])
                    print(data.zs[:10, :])

                    print('Predictions:')
                    with torch.no_grad():
                        print(F.softmax(pred_pis_logits, dim=1)[:10, :])
                        print(pred_vals[:10, :])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        # make tournament to see whether the model has improved
        print(f"Epoch {g+1} evaluate model...")
        value_new_model_first = play_tournament(trained_model, data_generating_model, 25, game_start_state, mcts_simulations=100)
        print(f"Score First: {value_new_model_first}")
        value_new_model_second = 1- play_tournament(data_generating_model, trained_model, 25, game_start_state, mcts_simulations=100)
        print(f"Score Second: {value_new_model_second}")

        if 0.5*value_new_model_first + 0.5*value_new_model_second >= 0.55:
            if load_model_path is not None and new_arch and add_gen==gen:
                data_generating_model = ModelArchitecture()

            data_generating_model.load_state_dict(trained_model.state_dict())
            gen += 1
            save_model(trained_model, f"generation_{gen}.pt")


        #print(f"Generation {g}: Score First: {value_new_model_first}, Score Second: {value_new_model_second}")