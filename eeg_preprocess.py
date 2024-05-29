import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from queue import Queue
from scipy.signal import cheby2, cheb2ord, filtfilt
from scipy.fftpack import fft, fftfreq

def cheby2_bandpass_filter(data, wp, ws, gpass=3, gstop=60, rs=60, btype='bandpass', fs = 1000):
    wp = [wp[0] / (fs / 2), wp[1] / (fs / 2)]
    ws = [ws[0] / (fs / 2), ws[1] / (fs / 2)]
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed

def band_amplitude_calc(data, fs, band_ranges):
    ######################### fft部分重复，可模块化
    # 计算信号的FFT
    data_fft = fft(data)
    # 计算频率轴
    freqs = fftfreq(len(data), 1/fs)
    
    # 仅保留单边频谱
    data_fft = data_fft[:len(data_fft)//2]
    freqs = freqs[:len(freqs)//2]
    ###########################
    
    # 计算每个频段的总幅度
    band_amplitudes = {}
    for band_name, (low, high) in band_ranges.items():
        band_mask = (freqs >= low) & (freqs <= high) # Bool type 
        band_amplitudes[band_name] = np.sum(2 * np.abs(data_fft[band_mask]))
    
    return band_amplitudes

def process_data(data, fs, filter_params, band_ranges):
    filtered_data = cheby2_bandpass_filter(data, *filter_params, fs=fs)
    return band_amplitude_calc(filtered_data, fs, band_ranges)

def data_generator(queue, sampling_rate, n_iterations=10):
    """模拟持续不断的EEG数据接收"""
    for _ in range(n_iterations):
        data = np.random.randn(sampling_rate)  # 模拟1秒的EEG数据
        queue.put(data)
        time.sleep(1)  # 模拟1秒的采样间隔

def data_producer(queue, fs, n_iterations):
    data_generator(queue, fs, n_iterations)

def data_consumer(queue, fs, filter_params, band_ranges):
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(10):  # 限制任务数量，以便能够终止
            # 获取新的数据窗口
            data = queue.get()
            # 提交新任务到进程池
            future = executor.submit(process_data, data, fs, filter_params, band_ranges)
            futures.append(future)

            # 检查并获取已经完成的任务结果
            for f in as_completed(futures):
                result = f.result()
                print(result)
                futures.remove(f)
