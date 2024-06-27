
import numpy as np
from scipy.fftpack import fft, fftfreq

fs = 128
# Generate a 1x128 array of random numbers from a uniform distribution [0, 1)
sig = np.random.rand(1, 128)

L = len(sig) 
# 计算信号的FFT
sig_fft = fft(sig) 
# 计算频率轴
freqs = fftfreq(L, 1/fs)
print(len(freqs))

L_fft = len(sig_fft)

# 仅保留单边频谱
sig_fft = 2*np.abs(sig_fft[:L_fft//2])/L
freqs = freqs[:L_fft//2]

# get bounds of interest    
low = 1
high = 13

# get power in the band
# band_mask = (freqs >= low) & (freqs <= high) # Bool type 
index_low = np.argmin(np.abs(freqs - low))
print(index_low)
# freqs[low:high]
# sig_fft[index_low:index_high] 
# return np.sum(sig_fft[band_mask])

# from maze_game import maze_game 


# maze_game()
