import argparse
import os
import numpy as np 
import torch 

"""
Repeat each experiment for a number of times

"""
class NFQMain:
    def __init__(self, args):
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)

    def train(self):
        pass


def main(args):
    nfq = NFQMain(args)
    nfq.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")

    parser.add_argument("--data_dir", type=str, default='./Data')

    # argement with help text
    parser.add_argument("--epochs", type=int, default=300, help="Number of episodes to train for")
    parser.add_argument("--train_max_steps", type=int, default=250, help="Number of time-steps at each training episode")
    parser.add_argument("--test_max_steps", type=int, default=300, help="Number of time-steps at each test episode")

    
    main(parser.parse_args())