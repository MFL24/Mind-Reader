from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from process_data_new import find_ch_index

def plot_data(queue_plot):
    # refresh time
    t = 1
    # create  figure and axes
    fig = plt.figure()
    ax4 = fig.add_subplot(4,1,4)
    ax1 = fig.add_subplot(4,1,1,sharex=ax4)
    ax2 = fig.add_subplot(4,1,2,sharex=ax4)
    ax3 = fig.add_subplot(4,1,3,sharex=ax4)
    ax1.tick_params('x', labelbottom=False)
    ax2.tick_params('x', labelbottom=False)
    ax3.tick_params('x', labelbottom=False)

    while True:
        if not queue_plot.empty():
            # clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            # get data
            plot_data_tuple = queue_plot.get()
            # plot filter data
            ax1.plot(range(128*2),plot_data_tuple[find_ch_index('AF3'),:])
            ax2.plot(range(128*2),plot_data_tuple[find_ch_index('F4'),:])
            ax3.plot(range(128*2),plot_data_tuple[find_ch_index('T7'),:])
            ax4.plot(range(128*2),plot_data_tuple[find_ch_index('T8'),:])
            ax1.set_title('AF3',fontsize = 10)
            ax2.set_title('F4',fontsize = 10)
            ax3.set_title('T7',fontsize = 10)
            ax4.set_title('T8',fontsize = 10)
            # update axes
            plt.draw()
            # pause to check the plot
            plt.pause(t)