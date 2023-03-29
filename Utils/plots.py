"""
Use pandas or some new utility to calculate the moving average 

"""
import os 
import time 
import matplotlib.pyplot as plt
import seaborn as sns 
import seaborn as sns
import numpy as np

sns.set_palette(palette='viridis', n_colors=3)
xticks = np.arange(0,275,25)

class Plots:
    def __init__(self):
        self.folder_path = "Plots/" 
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    # TODO: replace this with something that is built into pandas
    def moving_average(self, r_array):
        m_avg =[]
        #Smoothing param for moving avg
        num_points=30
        for i in range(r_array.shape[0]):
            if i<num_points-1:
                # Make the start full
                if i<int((num_points-1)/2):
                    m_avg.append(np.mean(r_array[0:int((num_points-1)/2)]))
                else: 
                    m_avg.append(np.mean(r_array[int((num_points-1)/2):(num_points-1)]))
            else:
                current_sum = 0
                for j in range(num_points-1):
                    current_sum+=r_array[i-j]
                current_avg = current_sum/num_points
                m_avg.append(current_avg)
        m_avg = np.array(m_avg)
        return m_avg

    def plot_success(self, experiences, max_steps, epoch_no):
        success_path = self.folder_path + "success/"
        if not os.path.exists(success_path):
            os.makedirs(success_path)

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
        plt.savefig(success_path + f"success_{epoch_no}.png", bbox_inches='tight')
        plt.close()
    
    def plot_cost(self, all_learn_data, total_epochs):
        """
        Plot the cost of the last 10 epochs from all_learn_data
        """
        sns.set_palette(palette='magma', n_colors=3)
        cost_path = self.folder_path + "cost/"
        if not os.path.exists(cost_path):
            os.makedirs(cost_path)

        ep_cost_train = []
        count = 0
        collect = 0
        collect_last = []

        # collect cost per episode
        for item in all_learn_data:
            datum = item["episode"][1]
            ep_cost_train.append(datum)

            if count>=total_epochs-10:
                collect_last.append(datum)
                collect+= datum
            count+=1

        collect_last = np.array(collect_last)
        ep_cost_train = np.array(ep_cost_train)

        fig,ax = plt.subplots(1, figsize=(16,5), dpi = 100)
        
        xticks = np.arange(0, total_epochs+25, 25)
        yticks = np.arange(0.0,1.4,0.2)

        plt.xlim([0, total_epochs])
        plt.xticks(ticks=xticks,fontsize=14)
        plt.yticks(ticks=yticks,fontsize=14)

        ax.set_xlabel("Episode", fontsize=16, fontweight='bold')
        ax.set_ylabel("Training cost per episode", fontsize=16, fontweight='bold')
        ax.plot(ep_cost_train)
        
        m_avg = self.moving_average(ep_cost_train)
        ax.plot(m_avg)
        ax.legend(["Training cost","Moving average"], fontsize=16, loc='upper right')

        #plt.show()
        plt.savefig(cost_path + f"cost" + time.strftime("%Y%m%d_%H%M%S") + ".png", bbox_inches='tight')
        plt.close()

####### FINAL PLOTS IN THE PAPER ########