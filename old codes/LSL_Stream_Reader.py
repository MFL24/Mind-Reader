import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream

# Step 1: Resolve the stream
print("Looking for an LSL stream...")
streams = resolve_stream('type', 'EEG')

# Step 2: Create an inlet to read from the stream
inlet = StreamInlet(streams[0])

# Step 3: Read data from the inlet
print("Now reading from the stream...")
while True:
    # Get a new sample (you can also get timestamp optionally by calling inlet.pull_sample() )
    sample, timestamp = inlet.pull_sample()
    print(f"Timestamp: {timestamp}, Sample: {sample}")