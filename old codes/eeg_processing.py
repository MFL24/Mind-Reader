# eeg_processing.py
import numpy as np
from scipy.signal import cheby2, cheb2ord, filtfilt
from scipy.fftpack import fft, fftfreq

def cheby2_bandpass_filter(data, wp, ws, gpass=3, gstop=60, rs=60, btype='bandpass', fs=1000):
    wp = [wp[0] / (fs / 2), wp[1] / (fs / 2)]
    ws = [ws[0] / (fs / 2), ws[1] / (fs / 2)]
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed

def band_amplitude_calc(data, fs, band_ranges):
    data_fft = fft(data)
    freqs = fftfreq(len(data), 1/fs)
    data_fft = data_fft[:len(data_fft)//2]
    freqs = freqs[:len(freqs)//2]
    
    band_amplitudes = {}
    for band_name, (low, high) in band_ranges.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_amplitudes[band_name] = np.sum(2 * np.abs(data_fft[band_mask]))
    
    return band_amplitudes

def process_data(data, fs, filter_params, band_ranges):
    wp, ws = filter_params
    filtered_data = cheby2_bandpass_filter(data, wp, ws, fs=fs)
    amplitudes = band_amplitude_calc(filtered_data, fs, band_ranges)
    return amplitudes
