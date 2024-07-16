#!/usr/bin/env python
# coding: utf-8

from scipy.signal import cheb2ord, cheby2, filtfilt
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, fftfreq
from sklearn import svm
import joblib
import time

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

def extract_epoch(sig,epoch_num,select_ch = False):
    epoch_data = []
    number_epoch = max(epoch_num) 
    for i in range(number_epoch+1):
        index = np.where(epoch_num==i)[0]
        if select_ch:
            tempt = sig[:,index]
            epoch_data.append(tempt[select_ch,:])
        else:
            epoch_data.append(sig[:,index])
    return epoch_data
    
def find_ch_index(ch):
    channel_list = np.array(['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'])
    try:
        index = np.where(channel_list==ch)[0][0]
    except:
        raise Exception ('Channel not found')
    return index

def fft_single_channel(data, fs, band_ranges, channel_index=False):
    if channel_index:
        sig = data[channel_index,:]
    else:
        sig = data
    L = len(sig)  # why 可以len(sig) sig不是numpy，需要.shape()吗？
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
    
    # get power in the band  sig_fft[band_mask] is a value
    band_mask = (freqs >= low) & (freqs <= high) # Bool type 
    
    # ！！！！！sig_fft[index_low:index_high] is a [13*1] vector, 保留更多信息
    # index_low = np.argmin(np.abs(freqs - low)) # or direct use find() 
    # index_high = np.argmin(np.abs(freqs - high)) # 鲁棒性更好，如果时间窗是3s，那么fft横轴freq间隔是1/3
    # sig_fft[index_low:index_high]
    
    return np.sum(sig_fft[band_mask])


def window_feature_extraction(data, fs, band_ranges, select_channels):
    band_amplitudes = []
    for band_name in band_ranges.keys():
        channels = select_channels[band_name]
        if band_name == 'chew': # T7, T8 
            for ch in channels:
                band_amplitudes.append(fft_single_channel(data[find_ch_index(ch),:], fs, band_ranges[band_name]))
        elif band_name == 'blink': # AF3, F3, F4
            for ch in channels:
                band_amplitudes.append(fft_single_channel(data[find_ch_index(ch),:], fs, band_ranges[band_name]))   
        else:
            raise Exception('Action not found')     
    return np.array(band_amplitudes)

def predict_features(data, fs, filter_params, band_ranges, select_channels, model, mode='threshold'):
    # filtered_data = cheby2_bandpass_filter(data, fs, *filter_params)

    # filter
    lowpass_filter_params = filter_params[1]
    highpass_filter_params = filter_params[0]
    filtered_data = cheby2_lowpass_filter(data, fs, *lowpass_filter_params)
    filtered_data = cheby2_highpass_filter(filtered_data, fs, *highpass_filter_params)
    
    # feature extraction
    window_feature = window_feature_extraction(filtered_data, fs, band_ranges, select_channels)
    window_feature = window_feature.reshape(1,-1)
    
    if mode == 'random':
        action_predict = np.random.randint(0,4)
    elif mode == 'svm':
        # predict
        action_predict = model.predict(window_feature)   
        if np.mean(window_feature[0][3:]) > 200:
            action_predict = 3
    elif mode == 'threshold':
        action_predict = 0
        if window_feature[0][0] > 25:
            action_predict = 2
        elif window_feature[0][1] > 150:
            action_predict = 1
        if np.max(window_feature[0][2:]) > 200:
            action_predict = 3  
    elif mode == 'mix':
        action_predict = model.predict(window_feature) 
        if window_feature[0][0] > 120:
            action_predict = 2
        if np.mean(window_feature[0][3:]) > 200:
            action_predict = 3          
        
    if action_predict == 0:
        action = 'Still' # classType: NoneType, not str.
    elif action_predict == 1:   
        action = 'Left'
    elif action_predict == 2:   
        action = 'Right'
    elif action_predict == 3:   
        action = 'Up'
    
    # window_feature = [window_feature[0][0],window_feature[0][2]-window_feature[0][1],
    #                 window_feature[0][3],window_feature[0][4]]
        
    return (filtered_data,window_feature,action)


def process_data(queue_raw, queue_plot, queue_action, fs, filter_params, band_ranges, select_channels, mode):
    count = 0
    CheckNumber = 3
    # Initialize buffer list with fixed length 3
    AccuracyTest = []
    BufferCommand = [None]*CheckNumber
    loaded_model = joblib.load("./NN_Model/SVM_Model_NoCHEWING.pkl")
    while True:
        if not queue_raw.empty():
            # 获取新的数据窗口
            raw_data_tuple = queue_raw.get()
            raw_data = raw_data_tuple[1] # type
            raw_time = raw_data_tuple[0]
            # 提交新任务到进程池
            result = predict_features(raw_data, fs, filter_params, band_ranges, select_channels, loaded_model, mode=mode)
            AccuracyTest.append(result[2])
            position = count%CheckNumber
            BufferCommand[position] = result[2]
            queue_plot.put(result[0])
            print(f'cycle: {count}')
            if BufferCommand[0] == BufferCommand[1] and BufferCommand[0] == BufferCommand[2]:
                print('Consistent')
                queue_action.put(result[2])                
            else:
                print('Not consistent')
                queue_action.put(0)
            # print(f'features are {result[1]}')
            print(f'actions are {BufferCommand[0]},{BufferCommand[1]},{BufferCommand[2]}')
            print(f'features are: {result[1]}')
            # print(f'Accuracy test was {AccuracyTest}')

            print('-'*10)
            
            count += 1
            # print(f'actions are {result[2]}')
            # print('ready to plot')
            

            