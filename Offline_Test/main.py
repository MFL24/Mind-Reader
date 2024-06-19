from scipy.signal import cheb2ord, cheby2, filtfilt
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


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



data = pd.read_csv('./Offline_Test/TEST01.csv',sep=';')

time = data['Time:128Hz']
time = time.to_numpy()
T8 = data['F4']
T8 = T8.to_numpy()

fs = 128
time = np.arange(0,1,1/fs)
tempt = []

for i in range(len(T8)):
    b = re.findall(r'\d*', T8[i])
    tempt.append(float(b[0]+b[2]+b[4]))

sig = np.array(tempt)

lowpass_filter_params = [27,28]
highpass_filter_params = [0.8,0.6]
filtered_data = cheby2_lowpass_filter(sig, fs, *lowpass_filter_params)
filtered_data = cheby2_highpass_filter(filtered_data, fs, *highpass_filter_params)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_axis_off()
ax.plot(filtered_data)
plt.show()


