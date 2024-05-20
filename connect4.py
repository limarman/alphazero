import copy
import numpy as np


class Connect4State:
    def __init__(self, position=None) -> None:
        self.rows = 6
        self.columns = 7
        self.action_space = list(range(self.columns))
        self.game_state = np.zeros((self.rows, self.columns)) if position is None else position
        self.finished = False
        self.draw = False

    def is_finished(self):
        return self.finished

    def get_possible_successors(self):
        actions = [col for col in self.action_space if self.game_state[0, col] == 0]
        succs = [copy.deepcopy(self).apply_action(action) for action in actions]
        return actions, succs

    def apply_action(self, action):
        if self.finished:
            raise ValueError("The game is already finished!")

        col = action
        if self.game_state[0, col] != 0:
            raise ValueError("Invalid move! The column is already full!")

        row = max([r for r in range(self.rows) if self.game_state[r, col] == 0])

        # apply move
        self.game_state[row, col] = 1

        # check whether this results in a win (or a draw)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # directions to check wins

        win = False
        for delta_x, delta_y in directions:
            number_of_ones = 1

            # checking the positive direction
            pos_x, pos_y = row + delta_x, col + delta_y
            while 0 <= pos_x < self.rows and 0 <= pos_y < self.columns:

                if self.game_state[pos_x, pos_y] == 1:
                    number_of_ones += 1
                else:
                    break

                pos_x += delta_x
                pos_y += delta_y

            # checking the negative direction
            pos_x, pos_y = row - delta_x, col - delta_y
            while 0 <= pos_x < self.rows and 0 <= pos_y < self.columns:

                if self.game_state[pos_x, pos_y] == 1:
                    number_of_ones += 1
                else:
                    break

                pos_x -= delta_x
                pos_y -= delta_y

            if number_of_ones >= 4:
                win = True
                break

        if win:
            self.finished = True
            self.draw = False
        elif np.all(self.game_state != 0):
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
    game = Connect4State()
    actions, succs = game.get_possible_successors()
    for succ in succs:
        print(succ.game_state)
