"""
Defines various exploration strategies
"""

import random
import numpy as np
from functools import partial

def get_action_with_probability(r, remaining):
    """
    Helper function to get action based on probability.
    """
    half = remaining / 2
    return 0 if r < half else 1

# epsilon-greedy exploration, linear decay (100% TO 5%)
def linear_ep_greedy(nfq_agent, ep, *args):
    r = random.random()

    # Linear decay percentage
    remaining = (episodes - ep) / episodes

    if r < remaining:
        # Take random action
        return get_action_with_probability(r, remaining)
    else:
        # Take definite action
        return nfq_agent.get_best_action(*args)

# epsilon-greedy exploration, exponential decay (100 TO 5%)
def exponential_ep_greedy(nfq_agent, ep, *args):
    r = random.random()

    # Exponential decay percentage
    remaining = np.exp(-0.015 * ep)

    if r < remaining:
        # Take random action
        return get_action_with_probability(r, remaining)
    else:
        # Take definite action
        return nfq_agent.get_best_action(*args)

# Epsilon greedy exploration with constant exploration at 2%
def constant_ep_greedy_two(nfq_agent, *args):
    r = random.random()
    
    if r < 0.02:
        # Take random action
        return get_action_with_probability(r, 1)
    return nfq_agent.get_best_action(*args)

# Epsilon greedy exploration with constant exploration at 10%
def constant_ep_greedy_ten(nfq_agent, *args):
    r = random.random()
    
    if r < 0.1:
        # Take random action
        return get_action_with_probability(r, 1)
    return nfq_agent.get_best_action(*args)

# No exploration
def no_exploration(nfq_agent, *args):
    return nfq_agent.get_best_action(*args)

def exploration_strategies(nfq_agent, strategy_name, ep=None):
    """
    Returns the specified exploration strategy with the `ep` argument
    passed only to the required functions.
    """
    strategies = {
        'linear': partial(linear_ep_greedy, nfq_agent, ep),
        'exponential': partial(exponential_ep_greedy, nfq_agent, ep),
        'constant_ten': partial(constant_ep_greedy_ten, nfq_agent),
        'constant_two': partial(constant_ep_greedy_two, nfq_agent),
        'no_exploration': partial(no_exploration, nfq_agent)
    }
    return strategies[strategy_name]


# Choose from
# linear, exponential, constant_ten, constant_two, no_exploration
