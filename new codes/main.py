from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from scipy.signal import cheb2ord, cheby2, filtfilt
from get_data import get_Data
from process_data import process_data
from plot_data import plot_data

if __name__ == '__main__':
    # set filter parameters
    fs = 128  # 采样率
    band_ranges = {
        # 'blink': (8, 13),
        'blink': (13, 30),
        'chew': (13, 30)
    }
    filter_params = ([5, 27], [4, 28])  # 示例滤波器参数
    channels = {
        'blink' : ('F3','F4'),
        'chew' : ('T7','T8')
    }
    thresholds = {
        'blink' : 0.1,
        'chew' : 0.1
    }

    # create 2 queues for data transfer, queue is shared between processes.
    queue_raw = mp.Queue()
    queue_plot = mp.Queue()
    
    # create 3 processes
    process_getData = mp.Process(target = get_Data, args=(queue_raw,))
    process_processData = mp.Process(target = process_data, args=(queue_raw, queue_plot, fs, filter_params, band_ranges, channels, thresholds))
    process_plotData = mp.Process(target=plot_data,args=(queue_plot,))

    process_getData.start()
    process_processData.start()
    process_plotData.start()
    
    process_getData.join()
    process_processData.join()
    process_plotData.join()