from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_data(queue_plot):
    # refresh time
    t = 3
    # create  figure and axes
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    while True:
        if not queue_plot.empty():
            # clear axes
            ax1.clear()
            ax2.clear()
            # get data
            plot_data_tuple = queue_plot.get()
            # plot raw an filter data
            ax1.plot(range(128*2),plot_data_tuple[0][2,:])
            # ax1.plot(range(128*2),plot_data_tuple[1][-3,:])
            ax2.plot(range(128*2),plot_data_tuple[1][2,:])
            ax1.set_title('F4')
            ax2.set_title('F3')
            # update axes
            plt.draw()
            # pause to check the plot
            plt.pause(t)