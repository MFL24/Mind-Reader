import numpy as np
import matplotlib.pyplot as plt
from CSV_to_dataset import CSV_to_Dataset
import os

if __name__ == '__main__':
    path = './DATA'
    fig, axes = plt.subplots(2,2, constrained_layout=True)
    valid_list = ['LeftEyeBlink4','RightEyeBlink6','Rest2']
    for root,_,file in os.walk(path):
        count = 0  
        for f in file: 
            if f in valid_list:
                filepath = os.path.join(root, f)
                a = CSV_to_Dataset(filepath,2,ERP=True)
                t = a[-1]
                axes.flat[count].plot(t,a[2][0,:])
                #axes.flat[count].fill_between(t,a[2][0,:] - a[3][0,:], a[2][0,:] + a[3][0,:], alpha=0.2)
                axes.flat[count].set_title(f+': F4')
                # axes.flat[count+9].plot(a[2][1,:])
                # axes.flat[count+9].set_title(f+': F3')
                count += 1
                
                axes.flat[count].plot(t,a[2][1,:])
                #axes.flat[count].fill_between(t,a[2][1,:] - a[3][1,:], a[2][1,:] + a[3][1,:], alpha=0.2)
                axes.flat[count].set_title(f+': AF3')
                
                count += 1
                
                axes.flat[count].plot(t,a[2][2,:])
                #axes.flat[count].fill_between(t,a[2][2,:] - a[3][2,:], a[2][2,:] + a[3][2,:], alpha=0.2)
                axes.flat[count].set_title(f+': T7')
                
                count += 1
                
                axes.flat[count].plot(t,a[2][3,:])
                #axes.flat[count].fill_between(t,a[2][3,:] - a[3][3,:], a[2][3,:] + a[3][3,:], alpha=0.2)
                axes.flat[count].set_title(f+': T8')            
                count += 1
    plt.show()



