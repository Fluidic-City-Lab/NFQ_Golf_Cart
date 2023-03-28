"""
Define 5 position initialization strategies in the range [-0.5, 0.5].
They are reported as variance and not std in the paper.
1. Gaussian Normal (mean, std) = (0, 0.15)
2. Gaussian Normal (mean, std) = (0, 0.3)
3. Uniform in the range
4. Linear increment
5. Exponential increment

Note: Epoch and episode are used interchangeably.
"""

import sys
import random
import torch 
import numpy as np 
from collections import namedtuple

# State definition
State = namedtuple('State', [
    "pos", # current position in fractions of a circle (0 = straight ahead)
    "vel", # velocity, i.e. the difference between the current and last position
    "voltage", # current voltage of the motor
               # (0 = stopped, 1 = full speed clockwise,
               #  -1 = full speed anti-clockwise)
])

class SteerboxEnv: 
    def __init__(self, env, env_type):
        self.state = None 
        self.env=env
        self.env_type=env_type

    # gaussian 1 
    def position_gaussian_1(self,epoch_no, epochs):
        return min(max(-0.5,random.gauss(0,0.15)),0.5) 
    
    # gaussian 2
    def position_gaussian_2(self,epoch_no, epochs):
        return min(max(-0.5,random.gauss(0,0.3)),0.5) 
    
    # uniformly at random
    def position_uniform(self,epoch_no, epochs):
        #return random.uniform(-0.5, 0.5) # same as below
        return ((2*random.random())-1)*0.5
    
    # curriculum linear
    def increment_position_linearly(self, epoch_no, epochs):
        epoch_no = epoch_no + 1 # starts at 0
        end_pos = 0.5 # Anywhere up to the position success
        increments = end_pos/epochs
        limit = increments * epoch_no
        return ((2*random.random())-1)*limit
    
    #curriculum exponential
    def increment_position_exponentially(self, epoch_no, epochs):
        epoch_no = epoch_no + 1 # starts at 0
        end_pos = 0.5 # Anywhere up to the position success
        limit = np.min([end_pos, np.exp(epoch_no)/(0.5*np.exp(200))])
        return ((2*random.random())-1)*limit
    
    #set the starting position of the wheel
    def reset(self, epoch_no, epochs):
        
        # Change this function to any of the 5 choices above, while performing experiments
        pos = self.position_uniform(epoch_no, epochs)
        #print("Start Position", pos)

        # initialize the current state
        self.state = State(pos, 0, 0)
        self.last_voltage = 0
        return np.array(self.state)
    
    def step(self, action):
        
        if self.env_type=='simulation':
            # perform the action in the environment and return the new state
            next_state = self.env.query_tree(self.state, action)
            # integrate the action directly as we can't trust the table to do it properly
            self.state = (next_state[0], next_state[1], self.last_voltage)
            
            # For simulation, change in voltate instead of actual voltage 
            dv = -0.1 if action == 0 else 0.1
            self.last_voltage = max(-1, min(1, self.last_voltage + dv))

            return np.array(self.state)

        else:
            # TODO: When implementing hardware add here
            print("Hardware not implemented yet")
            sys.exit(0) 
        
    def close(self):
        print("Closed")