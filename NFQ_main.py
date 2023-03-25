"""
## Five experiments: 
1. Parameter count of neural network
2. Size of Hint-to-goal transitions
3. Exploration strategy
4. Neural network reset frequence
5. Steering wheel position initialization

NFQ paper: https://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf
"""

import argparse
import os
import time 
import lzma 
import random

import numpy as np 
import torch 

from NFQ_Agent import NFQAgent
from Steer_Env import SteerBoxEnv
from NFQ_model import NFQNetwork
from NFQ_Env import Simulation

class NFQMain:
    def __init__(self, args):
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)

        
    def train(self):
        
        # generate unique seed for each experiment
        seeds = [random.randint(0, 1000000) for i in range(self.args.num_experiments)]
        
        # TODO: make it work with multiple experiments
        #for i in range(self.args.num_experiments):
        #print(f"Experiment: {i}, seed ={seeds[i]}")

        seed = seeds[0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.agent = NFQAgent(self.args)
        self.env = Simulation()
        #self.network =     

        self.env.build(self.args.data_dir)

        start = time.time()
        total_cost = 0

        for ep in range(1, self.args.episodes+1):
            print(f"Episode: {ep}")

            self.agent.train()
            #self.agent.evaluate()
            #self.agent.save_model()

        pass


def main(args):
    nfq = NFQMain(args)
    nfq.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Simulation", help="Choose environment: Simulation or Real")
    parser.add_argument("--data_dir", type=str, default='./Data', help="Directory to store data hardware data, or laod data to build simulation")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run and average results")

    #parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to train for")
    parser.add_argument("--train_max_steps", type=int, default=250, help="Number of time-steps at each training episode")
    parser.add_argument("--test_max_steps", type=int, default=300, help="Number of time-steps at each test episode")
    main(parser.parse_args())