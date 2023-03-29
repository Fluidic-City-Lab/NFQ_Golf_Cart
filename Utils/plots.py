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
        self.folder_path = "Plots/" + self.folder_name + "/"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def plot_success(self, experiences, max_steps, epoch_no):
        
        #print(experiences)
        success_plots_folder = "plots/success/"
        if not os.path.exists(success_plots_folder):
            os.makedirs(success_plots_folder)

        fig, ax = plt.subplots(1, figsize=(16,5), dpi = 100)
        ax.plot([e[0] for e in experiences])

        ax.legend(['Position', 'Velocity', 'Voltage'], fontsize=14)
        plt.xlim([0, max_steps])
        plt.ylim([-0.5, 0.5])
        plt.yticks(ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4,0.5], fontsize=14)

        plt.xticks(ticks=xticks,fontsize=14)
        plt.ylabel("Position range", fontsize=16, fontweight='bold')
        plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
        
        #plt.show()
        plt.savefig(success_plots_folder + f"success_{epoch_no}.png", bbox_inches='tight')
    pass 


####### FINAL PLOTS IN THE PAPER ########