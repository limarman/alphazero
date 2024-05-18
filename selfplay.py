from montecarlo import MonteCarloTree, DummyModel, TakeAwayState
import numpy as np
from model import SimpleTakeAwayModel

def self_play(start_state, model, mcts_simulations=500, temperature_tau=1, dirichlet_alpha=0.5):


    state = start_state

    states = []
    pis = []
    winners = []
    
    # initialize MCTS
    tree = MonteCarloTree(state=state, model=model, root_node=True, dirichlet_alpha=dirichlet_alpha)

    while not state.is_finished():
        
        states.append(state)

        # make MCT simulations
        for _ in range(mcts_simulations):
            tree.simulate(model=model)

        # choose next selfplay move
        actions, probabilities = get_actions_and_probabilities(tree.succ_dict, temperature_tau)
        next_action = actions[np.random.choice(len(actions), p=probabilities)]

        # add the prior labels
        pis.append({action: prob for action, prob in zip(actions, probabilities)})

        tree = tree.succ_dict[next_action]['succ']
        tree.promote_to_root_node(dirichlet_alpha, model)

        state = tree.state

    # compute the value labels - adjust for perspective switch (assuming alternating turns here!)
    z = state.get_final_value()
    winners = [1-z] * len(states)
    for i in range(len(states) - 2, 0, -2):
        winners[i] = z 
    
    states.append(state)

    return states, pis, winners


def get_actions_and_probabilities(successor_dict, temperature_tau):

    probabilities = np.array([entry['visit_count']**(1/temperature_tau) for _, entry in successor_dict.items()])
    probabilities /= np.sum(probabilities)
    actions = list(successor_dict.keys())

    return actions, probabilities


def play_tournament(first_model, second_model, no_games, start_state, temperature=1, mcts_simulations=500):

    total_return_model1 = 0

    for _ in range(no_games):
        
        moving_model = 0
        state = start_state

        while not state.is_finished():

            model_to_move = first_model if moving_model == 0 else second_model

            search_tree = MonteCarloTree(state=state, model=model_to_move, root_node=True)

            for _ in range(mcts_simulations):
                search_tree.simulate(model=model_to_move)
            
            actions, probabilities = get_actions_and_probabilities(search_tree.succ_dict, temperature)
            next_action = actions[np.random.choice(len(actions), p=probabilities)]

            state = search_tree.succ_dict[next_action]['succ'].state

            # alternate turns
            moving_model = (moving_model + 1) % 2

        value = state.get_final_value()
        if moving_model == 0:
            total_return_model1 += value
        else:
            total_return_model1 += (1-value)

    return total_return_model1/no_games


if __name__ == '__main__':
    state = TakeAwayState(20)

    model = DummyModel()
    #model = SimpleTakeAwayModel()

    #states, pis, winners  = self_play(state, model)

    #for state, pi, winner in zip(states, pis, winners):
    #    print(state.match_no, pi, winner)

    value = play_tournament(model, model, 100, state)

    print("Playoff:")
    print(f"First Model {(value):.3f} : {(1- value):.3f} Second Model")