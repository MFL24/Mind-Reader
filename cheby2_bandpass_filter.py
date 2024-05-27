import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheb2ord, cheby2, filtfilt

def cheby2_bandpass_filter(data, wp, ws, gpass=3, gstop=60, rs=60, btype='bandpass', fs = 1000):
    wp = [wp[0] / (fs / 2), wp[1] / (fs / 2)]
    ws = [ws[0] / (fs / 2), ws[1] / (fs / 2)]
    N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
    b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器
    data_preprocessed = filtfilt(b, a, data)  # 使用滤波器对数据进行滤波
    return data_preprocessed

# 生成信号数据 替换为 实时采集的数据
fs = 1000  # 采样频率为 1000 Hz
t = np.linspace(0, 1, fs, endpoint=False)  # 1 秒钟内的采样点
data = np.sin(50.0 * 2.0*np.pi*t) + 0.5*np.sin(80.0 * 2.0*np.pi*t)  # 信号数据

# 对信号进行滤波处理
data_preprocessed = cheby2_bandpass_filter(data, wp=[40, 60], ws=[30, 70])

# 绘制原始信号和滤波后的信号
plt.figure(figsize=(10, 6))
plt.plot(t, data, label='Original Signal', color='blue')
plt.plot(t, data_preprocessed, label='Filtered Signal', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original and Filtered Signals')
plt.legend()
plt.grid(True)
plt.show()
