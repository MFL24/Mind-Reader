import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne.preprocessing import ICA, corrmap
from scipy.signal import cheb2ord, cheby2, filtfilt

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

# path = './DATA/RightEyeBlink6'
# path = './DATA/LeftEyeBlink4'
path = './DATA/Chewing2'

data = pd.read_csv(path,sep=',')

epoch_number = data['Epoch']
epoch_number = epoch_number.to_numpy()

fs = 128

sig = data[data.keys()[2:-3]]
sig = sig.to_numpy().T

channel_list = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
info = mne.create_info(channel_list,128,'eeg')

epoch = extract_epoch(sig,epoch_number)

lowpass_filter_params = [27,28]
highpass_filter_params = [0.8,0.6]

filtered_epoch = []
for i in epoch:
    filtered_data = cheby2_lowpass_filter(i, fs, *lowpass_filter_params)
    filtered_data = cheby2_highpass_filter(filtered_data, fs, *highpass_filter_params)
    filtered_epoch.append(filtered_data)

epoch = np.reshape(filtered_epoch,(-1,14,128))

event = np.zeros((epoch.shape[0],3))
event[:,-1] = 1

epoch = mne.EpochsArray(epoch,info)

builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
montage = mne.channels.make_standard_montage("standard_1020")
epoch.set_montage(montage)

ica = ICA(n_components=10, max_iter="auto", random_state=97)
ica.fit(epoch)
# ica.plot_components()

ica.plot_properties(epoch, picks=[0, 3])



