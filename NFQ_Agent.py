import torch 
import torch.optim as optim

class NFQAgent:
    def __init__(self, net):
        self.net = net 
        self.optimizer = optim.Rprop(self.net.parameters()) # Rprop is the default for NFQ
        
    def get_best_action(self, state):
        """
        Make copies of the network and evaluate Q-value for each (state, action) combination
        Our controller is Bang-bang, can apply either 0 (0V) or 1 (5V)
        """
        # state, action= 0
        q_left = self.net(torch.cat([torch.FloatTensor(state), torch.FloatTensor([0])], dim=0))

        # state, action= 1
        q_right = self.net(torch.cat([torch.FloatTensor(state), torch.FloatTensor([1])], dim=0))
        
        # ...
        # Add more if more actions
        # ...

        # Lower Q value is better, return that
        return 1 if q_left >= q_right else 0

    def generate_pattern_set():
        """
        Update Q-values using stored transitions
        """
        pass

    def train(self, pattern_set):
        state_action_b, target_q_values = pattern_set
        pass

    def evaluate():
        pass
