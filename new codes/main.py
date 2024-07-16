from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from scipy.signal import cheb2ord, cheby2, filtfilt
from get_data import get_Data
# from process_data import process_data
from process_data_new import process_data
from plot_data import plot_data
from maze_game import maze_game

if __name__ == '__main__':
    # set filter parameters
    fs = 128  # 采样率#
    
    # set mode of classification, options are 'random', 'threshold', 'svm', 'mix'
    mode = 'threshold'
    
    band_ranges = {
        'blink': (1, 4),
        # 'blink': (13, 30),
        # 'chew': (1, 13),
        'chew': (13, 30)
    }
    # filter_params = ([0.8, 27], [0.6, 28])  # 示例滤波器参数
    filter_params = ([0.8, 0.6], [27, 28])
    
    # channels = {
    #     'blink' : (0,1,2),
    #     'chew' : (3,4)
    # }
    channels = {
        'blink' : ('F4','AF3'),
        # 'blink' : ('AF3','T7','T8'),
        'chew' : ('T7','T8')
    }

    
    thresholds = {
        'blink' : 50,
        'chew' : 105
    }

    # load model
    # clf = joblib.load("model.pkl")
    # print(clf.predict(X[0:1]))

    # create 2 queues for data transfer, queue is shared between processes.
    queue_raw = mp.Queue()
    queue_plot = mp.Queue()
    queue_action = mp.Queue()
    
    # create 3 processes
    process_getData = mp.Process(target = get_Data, args=(queue_raw,))
    process_processData = mp.Process(target = process_data, args=(queue_raw, queue_plot, queue_action, fs, filter_params, band_ranges, channels, mode))
    process_plotData = mp.Process(target=plot_data,args=(queue_plot,))
    process_mazeGame= mp.Process(target=maze_game,args=(queue_action,))

    process_getData.start()
    process_processData.start()
    process_plotData.start()
    process_mazeGame.start()
    
    process_getData.join()
    process_processData.join()
    process_plotData.join()
    process_mazeGame.join()