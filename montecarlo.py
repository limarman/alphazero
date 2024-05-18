import copy
import numpy as np

class MonteCarloTree:

    def __init__(self, state, parent=None, root_node=False, model=None, dirichlet_alpha=None) -> None:
        self.parent = parent
        self.state = state
        self.expanded = False
        self.network_value = None

        self.succ_dict = {}
        
        if parent is None and not root_node:
            raise ValueError("You either need to specifiy a parent node or make this node a root node!")

        if root_node:
            if model is None:
                raise ValueError("If you initialize as root node you need to also specify model")
            self.promote_to_root_node(dirichlet_alpha, model)

    #def promote_to_root_note(self):
    #    if not self.expanded:        

    def ucb_function(self, prior, visit_count):
        return prior/(1+visit_count)

    def promote_to_root_node(self, dirichlet_alpha, model):
        
        if not self.expanded:
            self.expand_node(model=model)

        self.parent = None

        # add dirichlet distribution to the current node
        rng = np.random.default_rng()  # Create a default random generator

        if len(self.succ_dict) > 0 and dirichlet_alpha is not None:
            alphas = [dirichlet_alpha] * len(self.succ_dict)

            dirichlet = rng.dirichlet(alpha = alphas)
            epsilon = 0.25

            for idx, key in enumerate(self.succ_dict.keys()):
                self.succ_dict[key]['prior'] = (1-epsilon) * self.succ_dict[key]['prior'] + epsilon * dirichlet[idx]

    def is_root_node(self):
        return self.parent is None

    def simulate(self, model):

        # follow the MCTS until a non-expanded node is reached
        # TODO: Handle terminal states of the game
        curr_node = self
        actions = []

        while curr_node.expanded and not curr_node.state.is_finished():
            action, curr_node = curr_node.choose_successor()
            actions.append(action)
        
        if not curr_node.expanded:
            curr_node.expand_node(model)
        
        leaf_val = curr_node.network_value
        

        # backup value back to the root
        curr_node = curr_node.parent
        curr_val = 1-leaf_val

        while not curr_node is None:
            action = actions.pop(-1)
            curr_node.succ_dict[action]['visit_count'] += 1
            curr_node.succ_dict[action]['cum_val'] += curr_val
            curr_node.succ_dict[action]['q_val'] = curr_node.succ_dict[action]['cum_val'] / curr_node.succ_dict[action]['visit_count']
            curr_node = curr_node.parent
            curr_val = 1-curr_val


    def choose_successor(self):
        if not self.expanded:
            raise ValueError("Node must the expanded first before it can choose a successor!")
        
        total_visits = sum(entry['visit_count'] for _, entry in self.succ_dict.items())

        # finding the maximal key
        max_succ = max(self.succ_dict, key=lambda x: self.succ_dict[x]['q_val'] + np.sqrt(total_visits) * self.ucb_function(self.succ_dict[x]['prior'], self.succ_dict[x]['visit_count']))

        return max_succ, self.succ_dict[max_succ]['succ']

    def expand_node(self, model):
        if self.expanded:
            raise ValueError("Node is already expanded!")            

        if self.state.is_finished():
            self.network_value = self.state.get_final_value()
        else:
            actions, succ_states = self.state.get_possible_successors()
            priors, self.network_value = model.evaluate(self.state)

            prob_sum = 0
            for action, next_state in zip(actions, succ_states):
                self.succ_dict[action] = {'q_val': 0, 'cum_val': 0, 'visit_count': 0, 'succ': MonteCarloTree(next_state, self), 'prior': priors[action]}
                prob_sum += priors[action]

            # normalize the prior distribution over the successors
            for succ in self.succ_dict:
                self.succ_dict[succ]['prior'] = self.succ_dict[succ]['prior'] / prob_sum

        self.expanded = True


class AsyncMonteCarloTree(MonteCarloTree):
    def __init__(self, state, task_queue, result_queue, ready_event, worker_id, parent=None, root_node=False, dirichlet_alpha=None) -> None:

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.ready_event = ready_event
        self.worker_id = worker_id

        super().__init__(state=state, parent=parent, root_node=root_node, model="NONE",
                         dirichlet_alpha=dirichlet_alpha)

    def expand_node(self, model):

        if self.expanded:
            raise ValueError("Node is already expanded!")

        if self.state.is_finished():
            self.network_value = self.state.get_final_value()
        else:
            actions, succ_states = self.state.get_possible_successors()
            #priors, self.network_value = model.evaluate(self.state)

            self.task_queue.put((self.worker_id, self.state))
            self.ready_event.clear()
            #print("Entered Wait...")
            self.ready_event.wait()
            #print("Exited Wait...")

            _, nn_output = self.result_queue.get()
            priors, self.network_value = nn_output

            prob_sum = 0
            for action, next_state in zip(actions, succ_states):
                self.succ_dict[action] = {'q_val': 0, 'cum_val': 0, 'visit_count': 0,
                                          'succ': AsyncMonteCarloTree(state=next_state, parent=self,
                                                                      task_queue=self.task_queue,
                                                                      result_queue=self.result_queue,
                                                                      ready_event=self.ready_event,
                                                                      worker_id=self.worker_id),
                                          'prior': priors[action]}
                prob_sum += priors[action]

            # normalize the prior distribution over the successors
            for succ in self.succ_dict:
                self.succ_dict[succ]['prior'] = self.succ_dict[succ]['prior'] / prob_sum

        self.expanded = True

class TakeAwayState:
    def __init__(self, match_no):
        self.match_no = match_no
        self.finished = match_no <= 0
        self.action_space = [1,2,3]
    
    def is_finished(self):
        return self.finished

    def get_possible_successors(self):

        if self.finished:
            return [], []

        actions = range(1,4)
        succs = [copy.deepcopy(self).apply_action(action) for action in actions]

        return actions, succs     

    def apply_action(self, action):
        
        if self.finished:
            raise ValueError("The game is already finished!")

        self.match_no -= action
        self.match_no = max(self.match_no, 0)

        self.finished = self.match_no == 0

        return self

    def get_final_value(self):
        if not self.finished:
            raise ValueError("The game is not over yet!")
        
        return 1
        

class DummyState:

    def __init__(self) -> None:
        self.val = 0
    
    def get_possible_successors(self):
        actions = range(0,2)
        succs = [copy.deepcopy(self).apply_action(action) for action in actions]

        return actions, succs

    def apply_action(self, action):
        if action == 0:
            self.val += 1
        if action == 1:
            self.val -= 1

        return self
            

class DummyModel:

    def __init__(self) -> None:
        pass

    def evaluate(self, state):
        # if state.val > 1:
        #     return {0: 0.5, 1: 0.5}, 1
        # else:
        #     return {0: 0.5, 1: 0.5}, 0
        return {1: 1/3, 2: 1/3, 3: 1/3}, 0.5


if __name__ == '__main__':
    #def ucb_func(prior, visit_count, q_val):
    #    return q_val + prior / (visit_count + 1)
    
    state = TakeAwayState(10)

    model = DummyModel()

    mct = MonteCarloTree(parent=None, state=state, root_node=True, model=model, dirichlet_alpha=0.9)

    for i in range(10000):
        mct.simulate(model=model)

    print(mct.succ_dict)