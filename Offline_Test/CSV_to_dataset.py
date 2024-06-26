from scipy.signal import cheb2ord, cheby2, filtfilt
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, fftfreq
from sklearn import svm

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


def band_feature_extraction(data, fs, band_ranges, select_channels):
    band_amplitudes = []
    for band_name in band_ranges.keys():
        channels = select_channels[band_name]
        if band_name == 'chew':
            for ch in channels:
                band_amplitudes.append(fft_single_channel(data[ch,:], fs, band_ranges[band_name]))
        elif band_name == 'blink':
            for ch in channels:
                band_amplitudes.append(fft_single_channel(data[ch,:], fs, band_ranges[band_name]))   
        else:
            raise Exception('Action not found')     
    return band_amplitudes


def CSV_to_Dataset(path,event_id):
    data = pd.read_csv(path,sep=';')
    
    epoch_number = data['Epoch']
    epoch_number = epoch_number.to_numpy()

    fs = 128

    sig = data[data.keys()[2::]]
    sig = sig.to_numpy().T

    ch_index = []
    channels = ['F4','F3','AF3','T7','T8']
    for ch in channels:
        ch_index.append(find_ch_index(ch))
        
    epoch = extract_epoch(sig,epoch_number,select_ch=ch_index)


    lowpass_filter_params = [27,28]
    highpass_filter_params = [0.8,0.6]
    
    filtered_epoch = []
    for i in epoch:
        filtered_data = cheby2_lowpass_filter(i, fs, *lowpass_filter_params)
        filtered_data = cheby2_highpass_filter(filtered_data, fs, *highpass_filter_params)
        filtered_epoch.append(filtered_data)

    _channels = {
        'blink' : (0,1,2),
        'chew' : (3,4)
    }

    band_ranges = {
        'blink': (1, 13),
        # 'blink': (13, 30),
        # 'chew': (1, 13),
        'chew': (13, 30)
    }

    feature_matrix = np.zeros((len(filtered_epoch),5))
    for i in range(len(filtered_epoch)):
        feature_matrix[i,:] = (band_feature_extraction(filtered_epoch[i],fs,band_ranges,_channels))
        
    label_vector = np.ones((len(filtered_epoch),1))*event_id

    return (feature_matrix,label_vector)

print(CSV_to_Dataset('./Offline_Test/RightEyeBlink3.csv',2)[0])

# model = svm.SVC() 
# model.fit()
# print(model.score())











































# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_axis_off()
# ax.plot(filtered_data[0,:])
# plt.show()


