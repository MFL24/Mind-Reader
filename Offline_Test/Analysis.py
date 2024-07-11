import numpy as np
import matplotlib.pyplot as plt
from CSV_to_dataset import CSV_to_Dataset
import os

if __name__ == '__main__':
    path = './DATA'
    analysis_type = 'FFT'
    if analysis_type == 'FFT':
        fig = plt.figure()
        ax = fig.add_subplot()
    elif analysis_type == 'ERP':
        fig, axes = plt.subplots(2,2, constrained_layout=True)
    valid_list = ['LeftEyeBlink4','RightEyeBlink6','Rest2','Chewing2']
    
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
                    axes.flat[count].set_ylabel('voltage (mV)')
                    count += 1
                elif analysis_type == 'FFT':
                    filepath = os.path.join(root, f)
                    a = CSV_to_Dataset(filepath,2,ERP=True)
                    band_mean = np.mean(a[0],axis=0)
                    band_std = np.std(a[0],axis=0)
                    ax.errorbar(['F4','AF3','T7','T8'],band_mean,yerr=band_std,
                                label=f[0:-1],capsize=10,capthick=1,marker='o')    
                    ax.set_ylabel('band power (mV)')
                    ax.set_xlabel('channel')
                    ax.legend()
    plt.show()