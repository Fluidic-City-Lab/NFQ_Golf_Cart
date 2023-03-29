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

import pickle
import numpy as np 
import torch 

from NFQ_Agent import NFQAgent
from NFQ_model import NFQNetwork
from Vehicle_Env import Simulation
from Steerbox_Env import SteerboxEnv
from Steerbox_NFQ import SteerboxNFQ 

from Utils.exploration_strategies import exploration_strategies

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
            data_file_name = "session_data/episode_"+time.strftime("%Y%m%d_%H%M%S")+".pickle.xz"

        else:
            # TODO: implement hardware environment
            print("Hardware environment not implemented in the main code base yet.")
            sys.exit()
        
        self.nfq_env = SteerboxNFQ(self.steer_env)
        self.nfq_agent = NFQAgent(self.args)   

        # Things that measure, collect
        start = time.time()
        total_cost = 0
        success_count = 0 

        all_learn_data = [] 
        all_experiences = [] 
        loss_total = [] 

        # Store hint-to-goal transitions (state, action) and q_value
        goal_state_action_b = []
        goal_target_q_values = []

        for ep in range(1, self.args.episodes+1):
            print(f"Episode: {ep}")

            # Which strategy to use for exploration
            exploration = exploration_strategies(self.nfq_agent, self.args.exploration, ep)


            # Perform an agent rollout
            success, new_experiences, episode_cost = self.nfq_env.experience(
                lambda *args: exploration(*args),
                self.args.train_max_steps,
                ep,
                self.args.episodes
            )
            success_count += success
            all_experiences.extend(new_experiences)
            total_cost += episode_cost

            # Generate the pattern set
            state_action_b, target_q_values = self.nfq_agent.generate_pattern_set(all_experiences)

            # hint-to-goal (% of total transitions), has to be calculated every time
            # only calculate how much to add i.e. the difference between desired and current
            new_size = int((1/100)*self.args.hint_size + 1) - len(goal_state_action_b)
            
            # Goal pattern set 
            new_goal_state_action_b, new_goal_target_q_values = self.nfq_env.generate_goal_pattern_set(size = new_size)
            goal_state_action_b.extend(new_goal_state_action_b)
            goal_target_q_values.extend(new_goal_target_q_values)

            # Convert to tensors
            t_goal_state_action_b = torch.FloatTensor(np.array(goal_state_action_b)) 
            t_goal_target_q_values = torch.FloatTensor(np.array(goal_target_q_values)) 

            # Attach hint-to-goal transitions
            state_action_b = torch.cat([state_action_b, t_goal_state_action_b], dim=0)
            target_q_values = torch.cat([target_q_values, t_goal_target_q_values], dim=0)

            # Hand over the current neural network
            old_agent = self.nfq_agent

            # Reset the Neural Network (Q-function approximator)
            if ep % self.args.reset_freq == 0:
                # Reset the weights
                print("\nResetting Network and Optimizer\n")
                self.nfq_agent = NFQAgent(self.args)
            
            # Train the agent
            loss_collection, last_step_loss = self.nfq_agent.train((state_action_b, target_q_values))
            loss_total.append(loss_collection)

            # DIABLE STAND-ALONE EVALUATION 
            # # Some metrics for evaluations 
            # num_evals = 0
            # eval_episode_length = 0
            # eval_episode_cost = 0

            # # Evaluate the agent 
            # while False: 
            #     eval_episode_length, eval_success, eval_episode_cost = nfq_agent.evaluate(nfq_env, EVAL_ENV_MAX_STEPS, epoch, EPOCHS)
            #     if not eval_success: 
            #         break
            #     num_evals += 1


            # remember everything about this epoch
            all_learn_data.append({
                "epoch": ep, # index
                
                # length of episode, its total cost, and the loss of the last step
                "episode": (len(new_experiences), episode_cost, last_step_loss),

                # list of (state, action, cost, next_state, failed) tuples
                "all_experiences": all_experiences,
                
                # list of (*state, action) tuples given as input to the network
                "state_action_b": np.asarray(state_action_b),
                
                # the Q function values the network should learn
                "target_q_values": np.asarray(target_q_values),
                
                # the network that generated the above values and ran this episode
                "net_state": old_agent.net.state_dict(),
            })
            
            # At the end of the epsodes, save data
            # Saving will take time 
            if self.args.save_to_file:
                try:
                    p = pickle.dumps(all_learn_data)
                    save_path = f"./{self.args.env}/{data_file_name}" 
                    with lzma.open(save_path, "wb") as f:
                        f.write(p)
                    del p

                except KeyboardInterrupt:
                    # re-try the save if the user accidentally interrupted it
                    continue
                break

            end = time.time()
            print(f"\n\tTotal Time elapsed = {round((end - start), 2)} seconds")


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
    parser.add_argument("--num_params", type=int, default=171, help="Number of parameters to be learned, choose from 39, 61, 91, 121, 171")
    parser.add_argument("--exploration", type=str, default="exponential", help="Choose exploration strategy: linear, exponential, constant_ten, constant_two, no_exploration ")
    parser.add_argument("--hint_size", type=int, default=10, help="Size of hint-to-goal transitions. Choose from []")
    parser.add_argument("--reset_freq", type=int, default=50, help="Frequency of resetting the Neural Network (Q-function approximator). Choose from  []")

    main(parser.parse_args())

# TODO: Save the terminal output to a log file, present in regressor code