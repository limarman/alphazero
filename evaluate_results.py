import numpy as np
import torch

from connect4 import Connect4State
from montecarlo import MonteCarloTree, TakeAwayState
from selfplay import get_actions_and_probabilities
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
    model = load_model("Connect4/buffer10k/generation_39.pt")

    turn = 0

    while not game_state.is_finished():

        print(f"State: \n {game_state.game_state}")

        if turn == 0:
            mct = MonteCarloTree(parent=None, state=game_state, root_node=True, model=model, dirichlet_alpha=None)

            for i in range(3000):
                mct.simulate(model=model)

            actions, probs = get_actions_and_probabilities(mct.succ_dict, temperature_tau=0)
            action = actions[np.random.choice(len(actions), p=probs)]

            #print(action, probs)

            print(f"MCT:")
            print(f"Expected Value: {mct.succ_dict[action]['q_val']}")
            for key, item in mct.succ_dict.items():
                print(f"{key}: Prior: {item['prior']}, visits {item['visit_count']}")

            print(f"Model chooses action {model.map_actionname_to_index(action)}.")
            game_state.apply_action(action)

            turn = 1

        elif turn == 1:
            while(True):
                action = input("Give your action as index (0-6): ")
                try:
                    game_state.apply_action(model.map_index_to_actionname(int(action)))
                    break
                except Exception:
                    print("This was an invalid move! Try again.")

            turn = 0


    print(f"FINAL STATE: \n {game_state.game_state}")
