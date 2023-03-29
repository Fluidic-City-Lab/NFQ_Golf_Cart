"""
Defines the Agent
"""
import torch 
import random
import torch.optim as optim
import torch.nn as nn
import numpy as np 

from NFQ_model import NFQNetwork

class NFQAgent:
    def __init__(self, args):
        
        self.args = args
        self.net = NFQNetwork(self.args.num_params) 
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

    def generate_pattern_set(self, experiences):
        """
        Pattern set = supervised dataset from transitions
        """

        states, actions, costs, next_states, dones = zip(*experiences)
        
        # b means batch
        #state_b = torch.FloatTensor(states) # This is slow
        state_b = torch.FloatTensor(np.array(states))
        action_b = torch.FloatTensor(actions)
        cost_b = torch.FloatTensor(costs)
        #next_state_b = torch.FloatTensor(next_states) # This is slow
        next_state_b = torch.FloatTensor(np.array(next_states))
        done_b = torch.FloatTensor(dones)

        state_action_b = torch.cat([state_b, action_b.unsqueeze(1)], 1)
        assert state_action_b.shape == (len(experiences), state_b.shape[1] + 1)

        # Current estimates of next state Q-values with actions = 0
        q_next_state_left_b = self.net(torch.cat([next_state_b, torch.zeros(len(experiences), 1)], 1)).squeeze()
        # Current estimates of next state Q-values with actions = 1
        q_next_state_right_b = self.net(torch.cat([next_state_b, torch.ones(len(experiences), 1)], 1)).squeeze()
        # Find the minimum (minimum is best) of the two
        q_next_state_b = torch.min(q_next_state_left_b, q_next_state_right_b)

        with torch.no_grad(): # TODO: remove this no grad?
            target_q_values = cost_b + self.args.gamma * q_next_state_b * (1 - done_b)

        # Return the supervised dataset
        return state_action_b, target_q_values

    def train(self, pattern_set):
        """
        Update Q-values using pattern set
        """
        # (State, action) and respective target Q-values in a batch
        state_action_b, target_q_values = pattern_set
        loss_collection = np.zeros(self.args.agent_epochs)

        for i in range(self.args.agent_epochs):
            predicted_q_values = self.net(state_action_b).squeeze()
            loss = nn.functional.mse_loss(predicted_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_collection[i] = loss.item()

        return np.array(loss_collection), loss.item()

    def evaluate(self, nfq_env, max_steps, epoch_no, epochs, pos_init):
        experiences, total_cost = nfq_env.experience(self.get_best_action, max_steps, epoch_no, epochs, pos_init)
        final_state = experiences[-1][3]

        success = (
            len(experiences) == max_steps
            and abs(final_state[0]) <= nfq_env.pos_success
            and abs(final_state[1]) <= nfq_env.vel_success
        )

        return len(experiences), success, total_cost



