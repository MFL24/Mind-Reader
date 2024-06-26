import mne
import numpy as np
import matplotlib.pyplot as plt


filepath = "C:/Wenlong Li/Master/2.Semester/Praktikum Biosignal/Recording/Test02.gdf"

data = mne.io.read_raw_gdf(filepath)

print(data)


