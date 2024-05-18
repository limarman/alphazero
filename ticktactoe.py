import copy
import numpy as np

class TicTacToeState:

    def __init__(self, position=None) -> None:
        self.action_space = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
        self.game_state = np.zeros((3,3)) if position is None else position
        self.finished = False
        self.draw = False

    def is_finished(self):
        return self.finished

    def get_possible_successors(self):
        
        places = np.where(self.game_state == 0)

        actions = list(zip(places[0], places[1]))
        succs = [copy.deepcopy(self).apply_action(action) for action in actions]

        return actions, succs
    
    def apply_action(self, action):
        if self.finished:
            raise ValueError("The game is already finished!")
        
        if self.game_state[action]  != 0:
            raise ValueError("Invalid move! The field is already occupied!")
        
        # apply move
        self.game_state[action] = 1

        # check whether this results in a win (or a draw)
        directions = [(1,1), (1,0), (0,1), (1,-1)]  # directions to check wins or loses

        win = False
        action_x, action_y = action
        for delta_x, delta_y in directions:
            number_of_ones = 1

            # checking the positive direction
            pos_x, pos_y =  action_x + delta_x, action_y + delta_y
            while 0 <= pos_x < 3 and 0 <= pos_y < 3:

                if self.game_state[pos_x, pos_y] == 1:
                    number_of_ones += 1

                pos_x += delta_x
                pos_y += delta_y

            # checking the negative direction
            pos_x, pos_y =  action_x - delta_x, action_y - delta_y
            while 0 <= pos_x < 3 and 0 <= pos_y < 3:

                if self.game_state[pos_x, pos_y] == 1:
                    number_of_ones += 1

                pos_x -= delta_x
                pos_y -= delta_y

            
            if number_of_ones == 3:
                win = True
                break

        if win:
            self.finished = True
            self.draw = False
        elif np.sum(self.game_state == 0) == 0:
            self.draw = True
            self.finished = True

        # flip the signs for the view point of the next player
        self.game_state = self.game_state * (-1)

        return self

    def get_final_value(self):
        if not self.is_finished():
            raise ValueError()
        
        if self.draw:
            return 0.5
        else:
            return 0


if __name__ == '__main__':
    game = TicTacToeState()

    #actions, succs = game.get_possible_successors()

    #print(succs)