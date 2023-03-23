"""
Major things defined here:
1. The reward function
2. The Hint-to-goal transitions 

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Utils import plots

class SteerboxNFQ:
    def __init__(self, env):
        self.env = env

        # NFQ relies on defined success (goal), forbidden (failure) and states between them
        self.pos_success = 0.05
        self.vel_success = 0.01
        self.pos_failure = 0.7
        self.vel_failure = 0.04

        # Minimum time control problem has a step cost
        # A penalty, when it neither succeeds nor fails
        self.step_cost = 0.001

    def reset(self, epoch_no, epochs):
        # Reset the environment and return the initial state
        return self.env.reset(epoch_no, epochs)
    
    def step(self, action):
        state = self.env.step(action)
        pos, vel, voltage = state

        # forbidden states
        if ((pos > self.pos_failure or pos < -self.pos_failure) or (vel > self.vel_failure or vel < -self.vel_failure)):
            failed = True
            cost = 1 # Very high cost

        # Goal states
        elif (-self.pos_success < pos < self.pos_success and -self.vel_success < vel < self.vel_success):
            failed = False
            cost = 0 # No cost

        # Neither
        else:
            failed = False
            cost = self.step_cost
            # The network is trying to rotate the wheel away from the center
            if (pos > 0 and voltage > 0) or (pos < 0 and voltage < 0): 
                cost *= 2 # Discourange this behavior with a higher than regular penalty
            
        return state, cost, failed
    
    def close(self):
        self.env.close()

    def experience(self, get_best_action, max_steps, epoch_no, epochs):
        state = self.reset(epoch_no, epochs)
        experiences = []
        
        total_cost = float(0.0)
        success_indicator = 0
        for step in range(max_steps):
            action = get_best_action(state)
            next_state, cost, failed = self.step(action)
                                    
            total_cost = total_cost + float(cost)
            experiences.append((state, action, cost, next_state, failed))
            #print("step:{} -> State(pos={:.4f}, vel={:.4f}, voltage={:.4f}), Cost= {}".format(step, *next_state, cost), end="\n")
            
            state = next_state
            if step ==249:
                #print("250:", state)
                if -0.05<next_state[0]<0.05 and -0.01<next_state[1]<0.01:
                    print("-------SUCCESS!!-------------")
                    success_indicator = 1
                print("step:{} -> State(pos={:.4f}, vel={:.4f}, voltage={:.4f}), Cost= {}".format(step, *next_state, cost), end="\n")
                
            if failed:
                break 
        
        if success_indicator ==1:
            plots.plot_success()
            
        return success_indicator, experiences, total_cost
    
    def generate_goal_pattern_set(self, size=200):
        """
        Artifically generate experiences in the region where the agent is likely to succeed, to help the network learn during early stages.
        Such transitions have a cost of 0
        """
        goal_state_action_b = [np.array([
            np.random.uniform(-self.pos_success, self.pos_success),
            np.random.uniform(-self.vel_success, self.vel_success),
            np.random.uniform(-0.2, 0.2), # change in voltage, this range is chosen  empirically
            np.random.randint(2), # Action at random
            ]) for _ in range(size)]

        goal_target_q_values = np.zeros(size)
        return goal_state_action_b, goal_target_q_values