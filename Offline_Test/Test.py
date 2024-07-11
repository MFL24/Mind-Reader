import numpy as np
import matplotlib.pyplot as plt
from CSV_to_dataset import CSV_to_Dataset
import os

if __name__ == '__main__':
    path = './DATA'
    fig, axes = plt.subplots(4,9, constrained_layout=True)
    for root,_,file in os.walk(path):
        count = 0
        for f in file:
            filepath = os.path.join(root, f)
            a = CSV_to_Dataset(filepath,2,ERP=True)
            axes.flat[count].plot(a[2][0,:])
            axes.flat[count].set_title(f+': F4')
            # axes.flat[count+9].plot(a[2][1,:])
            # axes.flat[count+9].set_title(f+': F3')
            axes.flat[count+9].plot(a[2][1,:])
            axes.flat[count+9].set_title(f+': AF3')
            axes.flat[count+9*2].plot(a[2][2,:])
            axes.flat[count+9*2].set_title(f+': T7')
            axes.flat[count+9*3].plot(a[2][3,:])
            axes.flat[count+9*3].set_title(f+': T8')            
            count += 1
    plt.show()



