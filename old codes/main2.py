import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading
import time

# 从独立模块导入函数
from eeg_preprocess import data_producer, data_consumer

fs = 128  # 采样率
band_ranges = {
    'blink': (6, 10),
    'chew': (18, 25)
}
filter_params = ([6, 25], [5, 26])  # 示例滤波器参数

data_queue = mp.Queue()

# 创建数据采集进程
producer_process = mp.Process(target=data_producer, args=(data_queue, fs, 10))
# 创建数据处理进程
consumer_process = mp.Process(target=data_consumer, args=(data_queue, fs, filter_params, band_ranges))

# 启动进程
producer_process.start()
consumer_process.start()

# 等待进程完成
producer_process.join()
consumer_process.join()
