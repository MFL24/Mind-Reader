#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from queue import Queue
from scipy.signal import cheby2, cheb2ord, filtfilt
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def cheby2_bandpass_filter(data, fs, wp, ws, gpass=3, gstop=40, rs=60, btype='bandpass'):
    wp = [wp[0] / (fs / 2), wp[1] / (fs / 2)]
    ws = [ws[0] / (fs / 2), ws[1] / (fs / 2)]
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed

def band_amplitude_calc(data, fs, band_ranges):
    ######################### fft部分重复，可模块化
    L = data.shape[1]
    # 计算信号的FFT
    data_fft = fft(data)
    # 计算频率轴
    freqs = fftfreq(len(data), 1/fs)
    
    # 仅保留单边频谱
    data_fft = 2*np.abs(data_fft[:len(data_fft)//2])/L
    freqs = freqs[:len(freqs)//2]
    ###########################
    
    # 计算每个频段的总幅度
    band_amplitudes = {}
    for band_name, (low, high) in band_ranges.items():
        band_mask = (freqs >= low) & (freqs <= high) # Bool type 
        band_amplitudes[band_name] = np.sum(2 * np.abs(data_fft[band_mask]))
    
    return band_amplitudes

def process(data, fs, filter_params, band_ranges):
    filtered_data = cheby2_bandpass_filter(data, fs, *filter_params)
    feature = band_amplitude_calc(filtered_data, fs, band_ranges)
    return (filtered_data,feature)

def process_data(queue_raw,queue_filter,queue_plot, fs, filter_params, band_ranges):

    while True:
        if not queue_raw.empty():
            # 获取新的数据窗口
            raw_data_tuple = queue_raw.get()
            raw_data = raw_data_tuple[1]
            raw_time = raw_data_tuple[0]
            # 提交新任务到进程池
            result = process(raw_data,fs, filter_params, band_ranges)
            print(result[1])
            queue_filter.put((raw_data,result[0]))
            print('ready to plot')
            queue_plot.put((raw_data,result[0]))
            # ax1.clear()
            # ax2.clear()
            # raw_curve = ax1.plot(range(128),raw_data[0,:])
            # filter_curve = ax2.plot(range(128),result[0][0,:])
            # plt.pause(0.5)
            # plt.show()
            