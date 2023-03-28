"""
## Five experiments: 
1. Parameter count of neural network
2. Size of Hint-to-goal transitions
3. Exploration strategy
4. Neural network reset frequence
5. Steering wheel position initialization

NFQ paper: https://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf
"""


import os
import sys
import time 
import lzma 
import random
import argparse

import numpy as np 
import torch 

from NFQ_Agent import NFQAgent
from NFQ_model import NFQNetwork
from NFQ_Env import Simulation
from Steerbox_Env import SteerboxEnv
from Steerbox_NFQ import SteerboxNFQ 

from utils.exploration_strategies import exploration_strategies

class NFQMain:
    def __init__(self, args):
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)

        
    def train(self):
        
        # generate unique seed for each experiment
        seeds = [random.randint(0, 1000000) for i in range(self.args.num_experiments)]
        
        # TODO: make it work on multiple (5) experiments at a time and average the results
        #for i in range(self.args.num_experiments):
        #print(f"Experiment: {i}, seed ={seeds[i]}")

        seed = seeds[0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize various heirarchies of environments
        if self.args.env == "Simulation":
            self.env = Simulation()
            self.steer_env = SteerboxEnv(self.env, env_type='simulation')
            self.env.build(self.args.data_dir)

            # Create folder to save sim data if required
            if self.args.save_to_file:
                if not os.path.exists("Simulation_Data"):
                    os.mkdir("Simulation_Data")

        else:
            # TODO: implement hardware environment
            print("Hardware environment not implemented yet.")
            sys.exit()
        
        self.nfq_env = SteerboxNFQ(self.steer_env)
        self.nfq_agent = NFQAgent(self.args)   

        # Things that measure, collect
        start = time.time()
        total_cost = 0
        success_count = 0 
        all_learn_data = [] 
        all_experiences = [] 

        exploration = exploration_strategies[self.args.exploration]

        for ep in range(1, self.args.episodes+1):
            print(f"Episode: {ep}")

            # Perform an agent rollout
            success, new_experiences, episode_cost = self.nfq_env.experience(exploration, self.args.train_max_steps, ep, self.args.episodes)
            success_count += success
            all_experiences.extend(new_experiences)
            total_cost += episode_cost

            # hint-to-goal (% of total transitions)
            size = 

            # Goal pattern set 


            # Reset the Neural Network (Q-function approximator)
            if ep % self.args.reset_freq == 0:
                # Reset the weights
                self.nfq_agent = NFQAgent(self.args)


        


def main(args):
    nfq = NFQMain(args)
    nfq.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Simulation", help="Choose environment: Simulation or Real")
    parser.add_argument("--data_dir", type=str, default='./Hardware_Data', help="Directory to store data hardware data, or laod data to build simulation")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run and average results")

    #parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to train for")
    parser.add_argument("--train_max_steps", type=int, default=250, help="Number of time-steps at each training episode")
    parser.add_argument("--test_max_steps", type=int, default=300, help="Number of time-steps at each test episode")
    
    ##
    parser.add_argument("--agent_epochs", type=int, default=150, help="How many training epochs of patter-set for agent training")
    parser.add_argument("--gamma", type=int, default=1.0, help="Discount factor")
    parser.add_argument("--save_to_file", type=bool, default=False, help="Save results to file")
    
    ## Related to experiments
    parser.add_argument("--exploration", type=str, default="epsilon_greedy", help="Choose exploration strategy:  XXX ")
    parser.add_argument("--hint_size", type=int, default=10, help="Size of hint-to-goal transitions. Choose from []")
    parser.add_argument("--reset_freq", type=int, default=50, help="Frequency of resetting the Neural Network (Q-function approximator). Choose from  []")

    main(parser.parse_args())

# TODO: Save the terminal output to a log file, present in regressor code