"""
Use pandas or some new utility to calculate the moving average 

"""
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_palette(palette='viridis', n_colors=3)
xticks = np.arange(0,275,25)

class Plots:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.folder_path = "plots/" + self.folder_name + "/"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        pass

    def plot_rewards(self, rewards, max_steps, xticks=xticks):
        fig,ax = plt.subplots(1, figsize=(16,5), dpi = 100)
        ax.plot(rewards)
        plt.xlim([0, max_steps])
        plt.ylim([-1, 1])
        plt.yticks(ticks=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1], fontsize=14)
        plt.xticks(ticks=xticks,fontsize=14)
        plt.ylabel("Reward", fontsize=16, fontweight='bold')
        plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
        #plt.show()
        plt.savefig(self.folder_path + "rewards.png", bbox_inches='tight')
        pass

    def plot_losses(self, losses, max_steps, xticks=xticks):
        fig,ax = plt.subplots(1, figsize=(16,5), dpi = 100)
        ax.plot(losses)
        plt.xlim([0, max_steps])
        plt.ylim([0, 0.5])
        plt.yticks(ticks=[0,0.1,0.2,0.3,0.4,0.5], fontsize=14)
        plt.xticks(ticks=xticks,fontsize=14)
        plt.ylabel("Loss", fontsize=16, fontweight='bold')
        plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
        #plt.show()
        plt.savefig(self.folder_path + "losses.png", bbox_inches='tight')
        pass

    # def plot_success(experiences, max_steps, xticks=xticks):

    # success_plots_folder = "plots/success/"
    # if not os.path.exists(success_plots_folder):
    #     os.makedirs(success_plots_folder)

    # fig,ax = plt.subplots(1, figsize=(16,5), dpi = 100)
    # ax.plot([e[0] for e in experiences])

    # ax.legend(['Position', 'Velocity', 'Voltage'], fontsize=14)
    # plt.xlim([0, max_steps])
    # plt.ylim([-0.5, 0.5])
    # plt.yticks(ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4,0.5], fontsize=14)

    # plt.xticks(ticks=xticks,fontsize=14)
    # plt.ylabel("Position range", fontsize=16, fontweight='bold')
    # plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
    
    # #plt.show()
    # plt.savefig(success_plots_folder + "success.png", bbox_inches='tight')
    # pass 



def plot_success(experiences, max_steps, xticks=xticks):

    success_plots_folder = "plots/success/"
    if not os.path.exists(success_plots_folder):
        os.makedirs(success_plots_folder)

    fig,ax = plt.subplots(1, figsize=(16,5), dpi = 100)
    ax.plot([e[0] for e in experiences])

    ax.legend(['Position', 'Velocity', 'Voltage'], fontsize=14)
    plt.xlim([0, max_steps])
    plt.ylim([-0.5, 0.5])
    plt.yticks(ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4,0.5], fontsize=14)

    plt.xticks(ticks=xticks,fontsize=14)
    plt.ylabel("Position range", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
    
    #plt.show()
    plt.savefig(success_plots_folder + "success.png", bbox_inches='tight')
    pass 
