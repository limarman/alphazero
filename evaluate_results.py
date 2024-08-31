import copy

import numpy as np
import torch

from connect4 import Connect4State
from montecarlo import MonteCarloTree, TakeAwayState
from selfplay import get_actions_and_probabilities, play_tournament
from ticktactoe import TicTacToeState

def load_model(path):
    model = torch.load(path)
    return model

if __name__ == '__main__':

    #game_state = TicTacToeState()
    game_state = Connect4State()

    model = load_model("models/generation_10.pt")

    # arbitrary position
    game_state.game_state = np.array([[-0., -0., -0.,  0., -0.,  0., -0.],
                                    [-0.,  0., -0., -0., -0.,  0., -0.],
                                    [-0., 0.,  0.,  0., -0., -0., -0.],
                                    [-0.,  0., 0., 0., -0.,  0., -0.],
                                    [0., 0., 0.,  0., -0.,  0., -0.],
                                    [0.,  0.,  0., 1., 0., -0.,  -1.]])

    history = []
    moves = []

    turn = 1

    while not game_state.is_finished():

        print(f"State: \n {game_state.game_state}")
        history.append(game_state.game_state)

        if turn == 0:
            mct = MonteCarloTree(parent=None, state=game_state, root_node=True, model=model, dirichlet_alpha=None)

            for i in range(10000):
                mct.simulate(model=model)

            actions, probs = get_actions_and_probabilities(mct.succ_dict, temperature_tau=0)
            action = actions[np.random.choice(len(actions), p=probs)]

            print(f"MCT:")
            print(f"Expected Value: {mct.succ_dict[action]['q_val']}")
            for key, item in mct.succ_dict.items():
                print(f"{key}: Prior: {item['prior']}, visits {item['visit_count']}")

            print(f"Model chooses action {model.map_actionname_to_index(action)}.")
            history.append(game_state.game_state)
            moves.append(model.map_actionname_to_index(action))
            game_state.apply_action(action)

            turn = 1

        elif turn == 1:
            while(True):
                action = input("Give your action as index (0-6), or enter 'b' for takeback: ")
                try:
                    if action == "b":
                        if len(history) < 2:
                            raise ValueError()
                        game_state.game_state = history.pop()
                        game_state.game_state = history.pop()
                        break
                    else:
                        game_state.apply_action(model.map_index_to_actionname(int(action)))
                        turn = 0
                        break
                except Exception:
                    print("This was an invalid move! Try again.")

            moves.append(action)

    print(f"FINAL STATE: \n {game_state.game_state}")
    print(f"Moves: {moves}")
