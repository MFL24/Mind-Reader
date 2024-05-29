import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading
import time

# 从独立模块导入函数
from eeg_processing import process_data

# 设置进程启动方法
mp.set_start_method('spawn', force=True)

def data_generator(queue, sampling_rate):
    """模拟持续不断的EEG数据接收"""
    while True:
        data = np.random.randn(sampling_rate)  # 模拟1秒的EEG数据
        queue.put(data)
        time.sleep(1)  # 模拟1秒的采样间隔

def run_multiprocessing_task():
    fs = 128  # 采样率
    band_ranges = {
        'blink': (6, 10),
        'chew': (18, 25)
    }
    filter_params = ([6, 25], [5, 26])  # 示例滤波器参数

    data_queue = Queue()
    
    # 启动数据生成线程
    data_thread = threading.Thread(target=data_generator, args=(data_queue, fs))
    data_thread.daemon = True
    data_thread.start()

    with ProcessPoolExecutor() as executor:
        futures = []
        while True:
            # 获取新的数据窗口
            data = data_queue.get()
            # 提交新任务到进程池
            future = executor.submit(process_data, data, fs, filter_params, band_ranges)
            futures.append(future)

            # 检查并获取已经完成的任务结果
            for f in as_completed(futures):
                result = f.result()
                print(result)
                futures.remove(f)

# 在Jupyter Notebook中运行多进程任务
run_multiprocessing_task()
