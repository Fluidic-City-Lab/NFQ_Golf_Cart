import random
import torch 
import numpy as np 

"""
Define 5 position initialization strategies in the range [-0.5, 0.5].
They are reported as variance and not std in the paper.
1. Gaussian Normal (mean, std) = (0, 0.15)
2. Gaussian Normal (mean, std) = (0, 0.3)
3. Uniform in the range
4. Linear increment
5. Exponential increment
"""

class SteerBoxEnv: 
    def __init__(self):
        self.state = None 

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