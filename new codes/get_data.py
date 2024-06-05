#!/usr/bin/env python
# coding: utf-8

# In[3]:


def get_Data(queue):
    # set window parameters
    nSamples_Window = 128
    step = 64
    
    # start streaming 
    print("Looking for an LSL stream...")
    try: 
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        print("Connected to stream. Now receiving data...")
    except:
        raise ValueError('No Stream founded')

    # initialize arrays
    timestamp = np.zeros(nSamples_Window)
    sample = np.zeros((15,nSamples_Window))
    # initialize count variable for number of generated windows
    count = 0
    i = 0
    # Acquiring data stream from OpenVIBE
    while True:
        # Get channel data at certain timestamp
        sample_tempt, timestamp_tempt= inlet.pull_sample()
        # check if window is fully acquired
        if i%step== 0 and i!=0 and i!=step:
            count += 1 # a window is generated
            queue.put((timestamp,sample)) # put window data into queue
            sample = np.roll(sample,-step,axis=1) # shift the window for overlapping
            timestamp = np.roll(timestamp,-step,axis=1) # shift the time vector for overlapping
        # write stream data
        sample[:,i-step*count]=sample_tempt
        timestamp[:,i-step*count]=timestamp_tempt
        i += 1

