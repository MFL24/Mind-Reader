import numpy as np

def find_ch_index(ch):
    channel_list = np.array(['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'])
    try:
        index = np.where(channel_list==ch)[0][0]
    except:
        raise Exception ('Channel not found')
    return index

