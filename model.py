import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class SimpleTakeAwayModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc_feature1 = torch.nn.Linear(20, 16)
        self.activation = torch.nn.ReLU()
        self.fc_feature2 = torch.nn.Linear(16,16)
        self.fc_feature3 = torch.nn.Linear(16,16)
        self.fc_probs = torch.nn.Linear(16,3)
        self.fc_val = torch.nn.Linear(16,1)
        #self.softmax = torch.nn.Softmax(dim=1)
        self.squash = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc_feature1(x))
        x = self.activation(self.fc_feature2(x))
        x = self.activation(self.fc_feature3(x))
        probs_logits = self.fc_probs(x)
        #probs = self.softmax(probs)
        val = self.fc_val(x)
        val = self.squash(val)

        return probs_logits, val
    
    def evaluate(self, single_state):

        with torch.no_grad():
            state_representation = self.construct_state_representation(state=single_state)
            tensor = torch.tensor(state_representation).unsqueeze(0)
            probs_logits, val = self.forward(tensor)
            probs = F.softmax(probs_logits, dim=1)

        prob_dict = {1: probs.squeeze()[0], 2: probs.squeeze(0)[1], 3:probs.squeeze(0)[2]}

        return prob_dict, val.item()
    
    #def construct_tensor(self, state):
    #    return torch.tensor([state.match_no], dtype=torch.float32).unsqueeze(0)
    
    def construct_state_representation(self, state):
        state_representation = np.zeros((20), dtype=np.float32)
        state_representation[state.match_no-1] = 1
        return state_representation

    def map_actionname_to_index(self, actionname):
        return actionname-1
    

class TicTacToeModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torch.nn.Conv2d(2, 8, kernel_size=(3,3), padding=1)
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=(3,3), padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, kernel_size=(3,3))

        self.fc_probs = torch.nn.Linear(8,9)
        self.fc_val = torch.nn.Linear(8,1)
        self.squash = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        probs_logits = self.fc_probs(x)
        #probs = self.softmax(probs)
        val = self.fc_val(x)
        val = self.squash(val)

        return probs_logits, val

    def evaluate(self, single_state):

        with torch.no_grad():
            state_representation = self.construct_state_representation(state=single_state)
            tensor = torch.tensor(state_representation).unsqueeze(0)
            probs_logits, val = self.forward(tensor)
            probs = F.softmax(probs_logits, dim=1)

        #prob_dict = {1: probs.squeeze()[0], 2: probs.squeeze(0)[1], 3:probs.squeeze(0)[2]}

        prob_dict = {self.map_index_to_actionname(index): probs.squeeze(0)[index] for index in range(probs.shape[1])}

        return prob_dict, val.item()

    def construct_state_representation(self, state):
        own_pieces = (state.game_state == 1).astype(np.float32)
        opponent_pieces = (state.game_state == -1).astype(np.float32)

        state_repr = np.stack([own_pieces, opponent_pieces])

        return state_repr

    def map_actionname_to_index(self, actionname):
        return actionname[0]*3 + actionname[1]
    
    def map_index_to_actionname(self, index):
        return (index // 3, index % 3)


class Connect4Model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torch.nn.Conv2d(2, 16, kernel_size=(3, 3), padding=1)
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3))

        self.fc_probs = torch.nn.Linear(16*2*3, 7)
        self.fc_val = torch.nn.Linear(16*2*3, 1)
        self.squash = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        probs_logits = self.fc_probs(x)
        # probs = self.softmax(probs)
        val = self.fc_val(x)
        val = self.squash(val)

        return probs_logits, val

    def evaluate(self, single_state):
        with torch.no_grad():
            state_representation = self.construct_state_representation(state=single_state)
            tensor = torch.tensor(state_representation).unsqueeze(0)
            probs_logits, val = self.forward(tensor)
            probs = F.softmax(probs_logits, dim=1)

        # prob_dict = {1: probs.squeeze()[0], 2: probs.squeeze(0)[1], 3:probs.squeeze(0)[2]}

        prob_dict = {self.map_index_to_actionname(index): probs.squeeze(0)[index] for index in range(probs.shape[1])}

        return prob_dict, val.item()

    def construct_state_representation(self, state):
        own_pieces = (state.game_state == 1).astype(np.float32)
        opponent_pieces = (state.game_state == -1).astype(np.float32)

        state_repr = np.stack([own_pieces, opponent_pieces])

        return state_repr

    def map_actionname_to_index(self, actionname):
        return actionname

    def map_index_to_actionname(self, index):
        return index
