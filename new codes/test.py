import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import find_channel_index
from scipy.signal import cheby2, cheb2ord, filtfilt,  bode, freqz
import pygame
from maze_game import maze_game 
import multiprocessing as mp
import time
from ..Offline_Test.CSV_to_dataset import CSV_to_Dataset
# from Offline_Test import CSV_to_dataset


feature_LeftEyeBlink1, label_LeftEyeBlink1 = CSV_to_Dataset('./DATA/LeftEyeBlink4', 1)
feature_LeftEyeBlink2, label_LeftEyeBlink2 = CSV_to_Dataset('./DATA/LeftEyeBlink2', 1)
feature_LeftEyeBlink3, label_LeftEyeBlink3 = CSV_to_Dataset('./DATA/LeftEyeBlink3', 1)

feature_RightEyeBlink1, label_RightEyeBlink1 = CSV_to_Dataset('./DATA/RightEyeBlink6', 2)
feature_RightEyeBlink2, label_RightEyeBlink2 = CSV_to_Dataset('./DATA/RightEyeBlink7', 2)

feature_Chewing1, label_Chewing1 = CSV_to_Dataset('./DATA/Chewing2', 3)
feature_Chewing2, label_Chewing2 = CSV_to_Dataset('./DATA/Chewing3', 3)

feature_Rest1, label_Rest1 = CSV_to_Dataset('./DATA/Rest2', 0)
feature_Rest2, label_Rest2 = CSV_to_Dataset('./DATA/Rest3', 0)



features = np.concatenate((feature_Rest1, feature_Rest2, feature_LeftEyeBlink1, feature_LeftEyeBlink2,
                        feature_LeftEyeBlink3, feature_RightEyeBlink1, feature_RightEyeBlink2, feature_Chewing1, feature_Chewing2), axis=0)
labels = np.concatenate((label_Rest1, label_Rest2, label_LeftEyeBlink1, label_LeftEyeBlink2,
                        label_LeftEyeBlink3, label_RightEyeBlink1, label_RightEyeBlink2, label_Chewing1, label_Chewing2), axis=0)