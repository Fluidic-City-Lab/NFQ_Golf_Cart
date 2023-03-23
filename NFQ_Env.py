"""
Contains 2 parts: 
1. Simulation: 
    What is a RL (simulation) environment?
        a. A trainsition function
        b. A reward function

    The transition function is built upon the KNN algorithm (with k=1) applied to the data collected form hardware.
        Step 1: For a given state, we find the nearest neighbor
        Step 2: We transition to the same state as the nearest neighbor

    The reward function is defined in SteerBoxEnv
    
2. Hardware: TBD
"""
import os 
import lzma
import pickle
import numpy as np
from scipy.spatial import KDTree

class Simulation():
    """
    Query a state and action and receive next state
    """
    def __init__(self):

        self.transitions_0 = []
        self.transitions_1 = []

        self.action_zero_tree = None
        self.action_one_tree = None

    # Build the KNN model
    def build(self, directory):
        """
        Transitions are categorized by action (0,1) first and then by state.
        This halves the query/search time.
        """
        i = 0 # file counter
        print("Building Simulation...")

        for j in os.listdir(directory):
            print(f"File: {i+1}")
            j = f'{directory}/{j}'
            if j.endswith('.pickle.xz'):
                with lzma.open(j, "rb") as f:
                    file = pickle.load(f)
                    print("Total Epochs: ", len(file))

                    # Only load the last epoch
                    # Since this is real-time training data, the last epoch is where the hardware agent had learned best (most of the times)

                    last_item = file[-1]
                    #print(last_item)
                    for experience in last_item["all_experiences"]: 
                        # Each experience is a tuple of (State, Action, Reward, Next State)
                        # For our transition function, we only need (State, Action, Next State)
                        #print(experience, type(experience), type(experience[0]))

                        if experience[1] ==0:
                            self.transitions_0.append((experience[0], experience[1], experience[3]))
                        elif experience[1] ==1:
                            self.transitions_1.append((experience[0], experience[1], experience[3]))
                        else:
                            raise ValueError("Action must be 0 or 1")
                    print("Transitions added from this file: ", len(last_item["all_experiences"]))
                    print("................................")
            i += 1
        print("Total Transitions Collected on Action 0:", len(self.transitions_0))
        print("Total Transitions Collected on Action 1:", len(self.transitions_1))
        print("................................")

        # List of query states, TODO: shorten the code?
        transitions_zero_query_states = np.array([item[0] for item in self.transitions_0])
        transitions_one_query_states = np.array([item[0] for item in self.transitions_1])
        print(f"Before tree: {transitions_zero_query_states.shape, transitions_one_query_states.shape}")
              
        # Query trees
        self.action_zero_tree = KDTree(transitions_zero_query_states) 
        self.action_one_tree = KDTree(transitions_one_query_states) 
        print(f"After tree: {self.action_zero_tree.data.shape, self.action_one_tree.data.shape}")
        print("................................")

    # Query the KNN model
    def query(self, state, action):
        if action == 0:
            # The index returned here is index of transitions zero
            dist, ind = self.action_zero_tree.query(state, k=1)
            #print("distance to neighbor:", dist )
            next_state = self.transitions_0[ind][2]
        
        elif action == 1:
            dist, ind = self.action_one_tree.query(state, k=1) 
            #print("distance to neighbor:", dist )
            next_state = self.transitions_1[ind][2]

        else:
            raise ValueError("Action must be 0 or 1")
            next_state = 0 

        return next_state

# Use case:
# sim = Simulation()
# sim.build('./Data')
# next_state = sim.query(np.array([ 0.0073162 , -0.00169522, -0.3       ]), 0)
# print(f"\nNext state: {next_state}\n")

# TODO: Interface the Hardware Environment here
# The hardware has its own code ATM.