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
from find_channel_index import find_ch_index

def cheby2_bandpass_filter(data, fs, wp, ws, gpass=3, gstop=40, rs=60, btype='bandpass'):
    wp = [wp[0] / (fs / 2), wp[1] / (fs / 2)]
    ws = [ws[0] / (fs / 2), ws[1] / (fs / 2)]
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed

def cheby2_lowpass_filter(data, fs, wp, ws, gpass=3, gstop=40, rs=60, btype='lowpass'):
    wp = wp / (fs / 2)
    ws = ws / (fs / 2)
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed

def cheby2_highpass_filter(data, fs, wp, ws, gpass=3, gstop=40, rs=60, btype='highpass'):
    wp = wp / (fs / 2)
    ws = ws / (fs / 2)
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed


def fft_single_channel(data, fs, band_ranges, channel_index):
    sig = data[channel_index,:]
    L = len(sig)
    # 计算信号的FFT
    sig_fft = fft(sig)
    # 计算频率轴
    freqs = fftfreq(L, 1/fs)
    
    L_fft = len(sig_fft)
    # 仅保留单边频谱
    sig_fft = 2*np.abs(sig_fft[:L_fft//2])/L
    freqs = freqs[:L_fft//2]
    
    # get bounds of interest    
    low = band_ranges[0]
    high = band_ranges[1]
    
    # get power in the band
    band_mask = (freqs >= low) & (freqs <= high) # Bool type 
    return np.sum(sig_fft[band_mask])


# def band_feature_extraction(data, fs, band_ranges, select_channels):
#     band_amplitudes = {}
#     for band_name in band_ranges.keys():
#         channels = select_channels[band_name]
#         if band_name == 'chew':
#             chew_feature = 0
#             for ch in channels:
#                 chew_feature = chew_feature + (fft_single_channel(data, fs, band_ranges[band_name], find_ch_index(ch)))
#             band_amplitudes[band_name] = chew_feature/len(channels)
#         elif band_name == 'blink':
#             blink_feature = []
#             for ch in channels:
#                 blink_feature.append(fft_single_channel(data, fs, band_ranges[band_name], find_ch_index(ch)))
#             band_amplitudes[band_name] = blink_feature    
#         else:
#             raise Exception('Action not found')     
#     return band_amplitudes

def feature_extraction(data, fs, band_ranges, select_channels):
    # band_amplitudes = {}
    for band_name in band_ranges.keys():
        channels = select_channels[band_name]
        if band_name == 'chew':
            chew_feature = 0
            for ch in channels:
                chew_feature = chew_feature + (fft_single_channel(data, fs, band_ranges[band_name], find_ch_index(ch)))
            band_amplitudes[band_name] = chew_feature/len(channels)
        elif band_name == 'blink':
            blink_feature = []
            for ch in channels:
                blink_feature.append(fft_single_channel(data, fs, band_ranges[band_name], find_ch_index(ch)))
            band_amplitudes[band_name] = blink_feature    
        else:
            raise Exception('Action not found')     
    return band_amplitudes

def predict_features(data, fs, filter_params, band_ranges, select_channels, thresholds):
    # filtered_data = cheby2_bandpass_filter(data, fs, *filter_params)
    # load model
    loaded_model = joblib.load("model.pkl")
    loaded_model.predict(data)
    lowpass_filter_params = filter_params[1]
    highpass_filter_params = filter_params[0]
    filtered_data = cheby2_lowpass_filter(data, fs, *lowpass_filter_params)
    filtered_data = cheby2_highpass_filter(filtered_data, fs, *highpass_filter_params)
    features = band_feature_extraction(filtered_data, fs, band_ranges,select_channels)

    action = 0

    # if features['blink'][0] > features['blink'][1] + thresholds['blink']:
    #     action = 2
    # elif features['blink'][1] > features['blink'][0] + thresholds['blink']:
    #     action = 3
        
    # if features['chew'] > thresholds['chew']:
    #     action = 1
    
    if features['blink'][0] > features['blink'][1] + thresholds['blink']:
        action = 'Left'
    elif features['blink'][1] > features['blink'][0] + thresholds['blink']:
        action = 'Right'
        
    if features['chew'] > thresholds['chew']:
        action = 'Up'
      
    return (filtered_data,features,action)



def process_data(queue_raw, queue_plot, queue_action, fs, filter_params, band_ranges, select_channels, thresholds):

    while True:
        if not queue_raw.empty():
            # 获取新的数据窗口
            raw_data_tuple = queue_raw.get()
            raw_data = raw_data_tuple[1]
            raw_time = raw_data_tuple[0]
            # 提交新任务到进程池
            result = predict_features(raw_data,fs, filter_params, band_ranges, select_channels, thresholds)
            print(f'features are {result[1]}')
            print(f'actions are {result[2]}')
            print('ready to plot')
            queue_plot.put((raw_data,result[0]))
            queue_action.put(result[2])
            # ax1.clear()
            # ax2.clear()
            # raw_curve = ax1.plot(range(128),raw_data[0,:])
            # filter_curve = ax2.plot(range(128),result[0][0,:])
            # plt.pause(0.5)
            # plt.show()
            