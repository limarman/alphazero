from typing import NamedTuple
import numpy as np
import torch
import os

from model import SimpleTakeAwayModel, TicTacToeModel
from montecarlo import TakeAwayState
from selfplay import self_play, play_tournament
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

if __name__ == '__main__':
    
    n_selfplays = 100
    n_gradient_steps = 5000 #10000
    n_iterations = 1
    n_epochs = 100

    buffer_size = 3000
    batch_size = 128

    game_start_state = TicTacToeState() #TakeAwayState(20)

    trained_model = TicTacToeModel() # SimpleTakeAwayModel()
    data_generating_model = TicTacToeModel() # SimpleTakeAwayModel()

    # initialize the same
    data_generating_model.load_state_dict(trained_model.state_dict())

    optimizer = torch.optim.Adam(params=trained_model.parameters(), lr=1e-3, weight_decay=1e-4)

    databuffer = Databuffer(buffer_size=buffer_size, state_shape=(2,3,3), action_shape=(9,))

    gen = 1

    for g in range(n_epochs):

        print(f"Generation {gen}")

        for i in range(n_iterations):
            
            if i % 100 == 0:
                print(f"Starting Iteration {i}...")

            for j in range(n_selfplays):
                states, pis, winners = self_play(game_start_state, data_generating_model, mcts_simulations=100, temperature_tau=1)

                for state, pi, z in zip(states, pis, winners):
                    
                    # generate the full pi for the agent (including moves that are not an option)
                    full_pi = np.zeros(len(game_start_state.action_space))

                    for key,value in pi.items():
                        full_pi[trained_model.map_actionname_to_index(key)] = value


                    databuffer.add_element(trained_model.construct_state_representation(state), full_pi, z)

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
        print(f"Epoch {g} evaluate model...")
        value_new_model_first = play_tournament(trained_model, data_generating_model, 50, game_start_state, mcts_simulations=100)
        print(f"Score First: {value_new_model_first}")
        value_new_model_second = 1- play_tournament(data_generating_model, trained_model, 50, game_start_state, mcts_simulations=100)
        print(f"Score Second: {value_new_model_second}")

        if 0.5*value_new_model_first + 0.5*value_new_model_second >= 0.55:
            data_generating_model.load_state_dict(trained_model.state_dict())
            gen += 1
            save_model(trained_model, f"generation_{gen}.pt")


        #print(f"Generation {g}: Score First: {value_new_model_first}, Score Second: {value_new_model_second}")