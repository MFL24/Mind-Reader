# get svm model
from scipy.signal import cheb2ord, cheby2, filtfilt
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, fftfreq
from sklearn import svm
import joblib
import torch
from torch.utils.data import Dataset,DataLoader

class EEG_Dataset(Dataset):
    def __init__(self,data,labels) -> None:
        super().__init__()
        self.data = data
        self.label = labels
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.data[index,:],self.label[index]
    

        
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
    
    #sig_fft[index_low:index_high] is a [13*1] vector, 保留更多信息
    # index_low = np.argmin(np.abs(freqs - low)) # or direct use find() 
    # index_high = np.argmin(np.abs(freqs - high)) # 鲁棒性更好，如果时间窗是3s，那么fft横轴freq间隔是1/3
    # sig_fft[index_low:index_high]
    
    return np.sum(sig_fft[band_mask])
    # return sig_fft[1:30]

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


def CSV_to_Dataset(path,event_id,ERP=False):
    data = pd.read_csv(path,sep=',')
    
    epoch_number = data['Epoch']
    epoch_number = epoch_number.to_numpy()

    fs = 128

    sig = data[data.keys()[2:15]]
    sig = sig.to_numpy().T

    ch_index = []
    # channels = ['F4','F3','AF3','T7','T8']
    channels = ['F4','AF3','T7','T8']
    for ch in channels:
        ch_index.append(find_ch_index(ch))
        
    epoch = extract_epoch(sig,epoch_number,select_ch=ch_index)


    lowpass_filter_params = [27,28]
    # lowpass_filter_params = [6,8]
    highpass_filter_params = [0.8,0.6]
    
    filtered_epoch = []
    for i in epoch:
        filtered_data = cheby2_lowpass_filter(i, fs, *lowpass_filter_params)
        filtered_data = cheby2_highpass_filter(filtered_data, fs, *highpass_filter_params)
        filtered_epoch.append(filtered_data)

    # _channels = {
    #     'blink' : (0,1,2),
    #     'chew' : (3,4)
    # }
    _channels = {
        'blink' : (0,1),
        # 'blink' : (1,2,3),
        'chew' : (2,3)
    }

    band_ranges = {
        'blink': (1, 4),
        # 'blink': (13, 39),
        'chew': (1, 4),
        # 'chew': (13, 30)
    }

    feature_matrix = np.zeros((len(filtered_epoch),len(channels)))
    for i in range(len(filtered_epoch)):
        feature_matrix[i,:] = (band_feature_extraction(filtered_epoch[i],fs,band_ranges,_channels))
        
    label_vector = np.ones((len(filtered_epoch),1))*event_id
    
    if ERP:
        filtered_epoch = np.array(filtered_epoch)
        ERP_mean = np.mean(filtered_epoch,axis=0)
        ERP_var = np.std(filtered_epoch,axis=0)
        t = np.arange(ERP_mean.shape[1])/fs
        return (feature_matrix,label_vector,ERP_mean,ERP_var,t)
    else:
        return (feature_matrix,label_vector)
        




feature_LeftEyeBlink1, label_LeftEyeBlink1 = CSV_to_Dataset('./DATA/LeftEyeBlink4', 1)
# feature_LeftEyeBlink2, label_LeftEyeBlink2 = CSV_to_Dataset('./DATA/LeftEyeBlink2', 1)
# feature_LeftEyeBlink3, label_LeftEyeBlink3 = CSV_to_Dataset('./DATA/LeftEyeBlink3', 1)

feature_RightEyeBlink1, label_RightEyeBlink1 = CSV_to_Dataset('./DATA/RightEyeBlink6', 2)
# feature_RightEyeBlink2, label_RightEyeBlink2 = CSV_to_Dataset('./DATA/RightEyeBlink7', 2)

# feature_Chewing1, label_Chewing1 = CSV_to_Dataset('./DATA/Chewing2', 3)
# feature_Chewing1, label_Chewing1 = CSV_to_Dataset('./DATA/Chewing3', 3)

feature_Rest1, label_Rest1 = CSV_to_Dataset('./DATA/Rest2', 0)
# feature_Rest2, label_Rest2 = CSV_to_Dataset('./DATA/Rest3', 0)

# print("Shape of feature_LeftEyeBlink:", feature_LeftEyeBlink.shape)
# print("Shape of feature_RightEyeBlink:", feature_RightEyeBlink.shape)
# print("Shape of feature_Chew:", feature_Chew.shape)

features = np.concatenate((feature_Rest1, feature_LeftEyeBlink1,
                        feature_RightEyeBlink1), axis=0)
labels = np.concatenate((label_Rest1,label_LeftEyeBlink1,
                        label_RightEyeBlink1), axis=0)


# features = np.concatenate((feature_Rest1, feature_Rest2, feature_LeftEyeBlink1, feature_LeftEyeBlink2,
#                            feature_LeftEyeBlink3, feature_RightEyeBlink1, feature_RightEyeBlink2, feature_Chewing1, feature_Chewing2), axis=0)
# labels = np.concatenate((label_Rest1, label_Rest2, label_LeftEyeBlink1, label_LeftEyeBlink2,
#                            label_LeftEyeBlink3, label_RightEyeBlink1, label_RightEyeBlink2, label_Chewing1, label_Chewing2), axis=0)

# dataset = EEG_Dataset(features,labels)

# train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)


# count = 1
# model = svm.SVC(kernel='rbf') 
# for train_feature, train_label in train_dataloader:
#     model.fit(train_feature, train_label)
#     print(model.score(train_feature,train_label))
#     count += 1

# from sklearn.model_selection import cross_val_score,ShuffleSplit
# cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# clf = svm.SVC(kernel='rbf')
# scores = cross_val_score(clf, features, labels, cv=cv)
# print(scores)


'''
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 0)

# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
model = svc.fit(X_train,y_train)

disp = ConfusionMatrixDisplay.from_estimator(
model,
X_test,
y_test,
display_labels=['Rest','LeftEye','RightEye'],
cmap=plt.cm.Blues,
normalize='true',
)



plt.show()


'''


# save model
# joblib.dump(model, "./NN_Model/SVM_Model_NoCHEWING.pkl")








































