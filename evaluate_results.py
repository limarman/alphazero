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

    #model = load_model("models/generation_5.pt")
    #game_state = TakeAwayState(12)

    #print(f"MODEL OUTPUT: {model.evaluate(game_state)}")


    #mct = MonteCarloTree(parent=None, state=game_state, root_node=True, model=model, dirichlet_alpha=None)

    #for i in range(100):
    #    mct.simulate(model=model)

    #print(f"MCT ROOT NODE: {mct.succ_dict}")

    # p = torch.tensor([0.2, 0.3, 0.5])

    # p_target = torch.tensor([0.2, 0.3, 0.5])

    # print(torch.nn.functional.cross_entropy(p, p_target))

    #game_state = TicTacToeState()
    game_state = Connect4State()

    #model = load_model("Connect4/buffer3000/generation_15.pt")
    #model = load_model("Connect4/buffer10k/generation_43.pt")
    #model = load_model("Connect4/buffer30k_addFC/generation_31.pt")
    #model = load_model("Connect4/buffer30k_4conv/generation_40.pt")
    #model = load_model("Connect4/buffer30k_64conv/generation_5.pt")
    model = load_model("models/generation_10.pt")

    #old_model = load_model("Connect4/buffer10k/generation_30.pt")

    #states = [copy.deepcopy(game_state).apply_action(i) for i in range(7)]

    #for idx, start_state in enumerate(states):
    #    print(f"Tournament Start for position {idx}...")
    #    print(f"Score as first: {play_tournament(model, old_model, start_state=start_state, no_games=1, mcts_simulations=100, temperature=0)}")
    #    print(f"Score as second: {1-play_tournament(old_model, model, start_state=start_state, no_games=1, mcts_simulations=100, temperature=0)}")
    #print(play_tournament(model, old_model, start_state=game_state, no_games=1, temperature=0, mcts_simulations=1))


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

            #print(action, probs)

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
