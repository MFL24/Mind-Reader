import numpy as np
import matplotlib.pyplot as plt
from CSV_to_dataset import CSV_to_Dataset
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


if __name__ == '__main__':
    path = './DATA'
    analysis_type = '3D'
    if analysis_type == 'FFT':
        fig = plt.figure()
        ax = fig.add_subplot()
    elif analysis_type == 'ERP':
        fig, axes = plt.subplots(2,2, constrained_layout=True)
    elif analysis_type == '3D':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    valid_list = ['LeftEyeBlink4','RightEyeBlink6','Rest2','Chewing2']
    id_list = {'LeftEyeBlink4':1,'RightEyeBlink6':2,'Rest2':0,'Chewing2':3}
    features = np.array([])
    labels = np.array([])
    
    for root,_,file in os.walk(path):
        count = 0  
        for f in file: 
            if f in valid_list:
                if analysis_type == 'ERP':
                    filepath = os.path.join(root, f)
                    a = CSV_to_Dataset(filepath,2,ERP=True)
                    t = a[-1]
                    axes.flat[count].plot(t,a[2][0,:],label='F4',color='#8ECFC9',
                                        ls='--'if f=='Chewing2' else '-')
                    #axes.flat[count].fill_between(t,a[2][0,:] - a[3][0,:], a[2][0,:] + a[3][0,:], alpha=0.2)

                    axes.flat[count].plot(t,a[2][1,:],label='AF3',color='#82B0D2',
                                        ls='--'if f=='Chewing2' else '-')
                    #axes.flat[count].fill_between(t,a[2][1,:] - a[3][1,:], a[2][1,:] + a[3][1,:], alpha=0.2)

                    axes.flat[count].plot(t,a[2][2,:],label='T7',color='#FA7F6F',
                                        ls='-'if f=='Chewing2' or f=='Rest2' else '--')
                    #axes.flat[count].fill_between(t,a[2][2,:] - a[3][2,:], a[2][2,:] + a[3][2,:], alpha=0.2)
                    
                    axes.flat[count].plot(t,a[2][3,:],label='T8',color='#FFBE7A',
                                        ls='-'if f=='Chewing2' or f=='Rest2' else '--')
                    #axes.flat[count].fill_between(t,a[2][3,:] - a[3][3,:], a[2][3,:] + a[3][3,:], alpha=0.2)
                    
                    axes.flat[count].legend()  
                    axes.flat[count].set_title(f[0:-1])
                    axes.flat[count].set_xlabel('time (s)')
                    axes.flat[count].set_ylabel('voltage (µV)')
                    count += 1
                elif analysis_type == 'FFT':
                    filepath = os.path.join(root, f)
                    a = CSV_to_Dataset(filepath,2,ERP=True)
                    band_mean = np.mean(a[0],axis=0)
                    band_std = np.std(a[0],axis=0)
                    ax.errorbar(['F4','AF3','T7','T8'],band_mean,yerr=band_std,
                                label=f[0:-1],capsize=10,capthick=1,marker='o')    
                    ax.set_ylabel('band power (µV)')
                    ax.set_xlabel('channel')
                    # ax.set_title('Band power between 13-30 Hz')
                    ax.set_title('Band power between 1-4 Hz')
                    ax.legend()
                elif analysis_type == 'Confusion':
                    filepath = os.path.join(root, f)
                    a = CSV_to_Dataset(filepath,id_list[f],ERP=False)
                    if count == 0:
                        features = a[0]
                        labels = a[1]
                    else:
                        features = np.concatenate((features,a[0]),axis=0)
                        labels = np.concatenate((labels,a[1]),axis=0)
                    if count == 3:
                        X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 0)
                        svc=SVC() 
                        model = svc.fit(X_train,y_train)
                        disp = ConfusionMatrixDisplay.from_estimator(
                        model,
                        X_test,
                        y_test,
                        #display_labels=class_names,
                        cmap=plt.cm.Blues,
                        normalize='true',
                        )
                    count += 1
                elif analysis_type == '3D':
                    filepath = os.path.join(root, f)
                    a = CSV_to_Dataset(filepath,2,ERP=True)         
                    points = np.zeros((a[0].shape[0],3))           
                    points[:,0:2] = a[0][:,0:2]
                    points[:,2] = np.mean(a[0][:,2],axis=0)
                    ax.scatter(points[:,0],points[:,1],points[:,2],label=f[0:-1])
                    ax.legend()
                    ax.set_xlabel('F4')
                    ax.set_ylabel('AF3')
                    ax.set_zlabel('(T7+T8)/2')

                    
    plt.show()