import argparse
import os
import numpy as np 
import torch 

class NFQMain:
    def __init__(self, args):
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)


def main(args):
    nfq = NFQMain(args)
    nfq.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    main(parser.parse_args())