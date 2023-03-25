"""
Use pandas or some new utility to calculate the moving average 

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_palette(palette='viridis', n_colors=3)
xticks = np.arange(0,275,25)

def plot_success():
    fig,ax = plt.subplots(1, figsize=(16,5), dpi = 100)
    ax.plot([e[0] for e in experiences])
    ax.legend(['Position', 'Velocity', 'Voltage'], fontsize=14)
    plt.xlim([0, max_steps])
    plt.ylim([-0.5, 0.5])
    plt.yticks(ticks=[-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4,0.5], fontsize=14)
    plt.xticks(ticks=xticks,fontsize=14)
    plt.ylabel("Position range", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps", fontsize=16, fontweight='bold')
    plt.show()
    pass 
