from CSV_to_dataset import CSV_to_Dataset
from scipy.signal import cheb2ord, cheby2, filtfilt
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, fftfreq
from sklearn import svm
import joblib
from sklearn.utils import shuffle
import random
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


def predict_features(window_feature, model, mode='threshold'):
    
    if mode == 'random':
        action_predict = np.random.randint(0,4)
    elif mode == 'svm':
        # predict
        action_predict = model.predict(window_feature)   
        if np.mean(window_feature[0][3:]) > 200:
            action_predict = 3
    elif mode == 'threshold':
        action_predict = 0
        if window_feature[0][0] > 50:
            action_predict = 2
        elif window_feature[0][1] > 150:
            action_predict = 1
        if np.max(window_feature[0][2:]) > 150:
            action_predict = 3  
    elif mode == 'mix':
        action_predict = model.predict(window_feature) 
        if window_feature[0][0] > 120:
            action_predict = 2
        if np.max(window_feature[0][3:]) > 200:
            action_predict = 3          
        
    if action_predict == 0:
        action = 0
    elif action_predict == 1:   
        action = 1
    elif action_predict == 2:   
        action = 2
    elif action_predict == 3:   
        action = 3
 
    # test_feature = [window_feature[0][0],window_feature[0][2]-window_feature[0][1],
    #                 window_feature[0][3],window_feature[0][4]] 
        
    return (window_feature,action)

            
if __name__ == '__main__':
    
    # load datasets
    # feature_LeftEyeBlink1, label_LeftEyeBlink1 = CSV_to_Dataset('./DATA/LeftEyeBlink4', 1)
    # feature_Chewing1, label_Chewing1 = CSV_to_Dataset('./DATA/Chewing2', 3)
    # feature_Rest1, label_Rest1 = CSV_to_Dataset('./DATA/Rest2', 0)
    features, labels = CSV_to_Dataset('./DATA/ThresholdAccuTest2', 1)
    
    # labels = [0, 0, 2, 0, 0, 1, 0, 1, 0, 2, 2, 1, 2, 3, 1, 3, 0, 3, 2, 1, 2, 
    #           3, 2, 0, 1, 0, 1, 2, 1, 0, 3]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2, 
    0, 0, 0, 1, 1, 1, 3, 3, 3]
    
    # fuse datasets
    # features = np.concatenate((feature_Rest1, feature_LeftEyeBlink1,
    #                         feature_RightEyeBlink1,feature_Chewing1), axis=0)
    # labels = np.concatenate((label_Rest1,label_LeftEyeBlink1,
    #                         label_RightEyeBlink1,label_Chewing1), axis=0) 
    
    # shuffle datasets
    # features, labels = shuffle(features, labels)
    
    # set filter parameters
    filter_params = ([0.8, 0.6], [27, 28])
    
    # loaded_model = joblib.load("./NN_Model/SVM_Model_NoCHEWING.pkl")
    loaded_model = joblib.load("./NN_Model/SVM_Model.pkl")
    thresholds = {
        'blink' : 70,
        'chew' : 105
    }
    # set stimulation cycles
    n_dataset = features.shape[0]
    print(n_dataset)
    predicted_action = []
    count = 0
    for i in range(n_dataset):
        picked_features = features[i,:]
        picked_features = np.reshape(picked_features,(1,-1))
        result = predict_features(picked_features, loaded_model, mode='svm')
        predicted_action.append(result[1])
        print(result[0])
        
    predicted_action = np.array(predicted_action)
    true_action = np.array(labels)
    print(predicted_action)
    print(np.sum(predicted_action==true_action)/n_dataset)
    print('-------')
    
    
    # cmatrix = confusion_matrix(true_action, predicted_action)
    # print(cmatrix)
    # disp = ConfusionMatrixDisplay(cmatrix,display_labels=['Rest','LeftEye','RightEye','Chewing'])
    # disp.plot()
    # plt.show()