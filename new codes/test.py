import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import find_channel_index
from scipy.signal import cheby2, cheb2ord, filtfilt,  bode, freqz

fs = 128
wp = 0.5
ws = 0.3
gpass=3
gstop=40
rs=60
btype='highpass'
wp = wp/(fs/2)
ws = ws/(fs/2)
N, Wn = cheb2ord(wp, ws, gpass, gstop)  # 计算order和归一化截止频率
b, a = cheby2(N, rs, Wn, btype)  # 设计Chebyshev II滤波器


w, h = freqz(b, a,fs=128)
plt.plot(w, 20 * np.log10(abs(h)))
plt.show()