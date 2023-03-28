"""
Defines various exploration strategies
"""

import random 
import numpy as np

#epsilon-greedy exploration, linear decay (100% TO 5%)
def linear_ep_greedy(nfq_agent, epoch, EPOCHS):
    r = random.random()
    
    # Linear decay percentage
    remaining = (EPOCHS - epoch)/EPOCHS

    if r < remaining:
        # Take random action
        half = remaining/2
        return 0 if r < half else 1
    
    elif remaining <= 0.05:
        # If less than 5% always take definite action
        return nfq_agent.get_best_action(*args)
    else: 
        # Otherwise take definite action
        return nfq_agent.get_best_action(*args)
    
    
#epsilon-greedy exploration, exponential decay (100 TO 5%)
def exponential_ep_greedy(nfq_agent, epoch):
    r = random.random()
    
    # Linear decay percentage
    remaining = np.exp(-0.015*epoch) #(EPOCHS - epoch)/EPOCHS
    #print(remaining)
    if r < remaining:
        # Take random action
        half = remaining/2
        return 0 if r < half else 1
    
    elif remaining <= 0.05:
        # If less than 5% always take definite action
        return nfq_agent.get_best_action(*args)
    else: 
        # Otherwise take definite action
        return nfq_agent.get_best_action(*args)

# Epsilon greedy exploration with constant exploration
def constant_ep_greedy(nfq_agent):   
    r = random.random()
    if r < 0.1:
        return 0 if r < 0.05 else 1
    return nfq_agent.get_best_action(*args)

exploration_strategies = {
    'linear': linear_ep_greedy,
    'exponential': exponential_ep_greedy,
    'constant': constant_ep_greedy
}
