import argparse
import os
import random
import numpy as np 
import torch 

from NFQ_Agent import NFQAgent
from Steer_Env import SteerBoxEnv
from NFQ_model import NFQNetwork

"""
Repeat each experiment for a number of times

"""

class NFQMain:
    def __init__(self, args):
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)

        

    def train(self):
        
        # generate unique seed for each experiment
        seeds = [random.randint(0, 1000000) for i in range(self.args.num_experiments)]
        for i in range(self.args.num_experiments):
            print(f"Experiment: {i}, seed ={seeds[i]}")
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

            # self.agent = NFQAgent(self.args)
            # self.agent.train()
   

        pass


def main(args):
    nfq = NFQMain(args)
    nfq.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Simulation", help="Choose environment: Simulation or Real")
    parser.add_argument("--data_dir", type=str, default='./Data', help="Directory to store data hardware data, or laod data to build simulation")
    parser.add_argument("--num_experiments", type=int, default=5, help="Number of experiments to run and average results")

    #parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=300, help="Number of episodes to train for")
    parser.add_argument("--train_max_steps", type=int, default=250, help="Number of time-steps at each training episode")
    parser.add_argument("--test_max_steps", type=int, default=300, help="Number of time-steps at each test episode")
    main(parser.parse_args())