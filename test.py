import time
import random

from multiprocessing import Process, Queue, Pool
from tqdm import tqdm
import numpy as np
from config import News20Config
import json
import six
import jieba
from spft.multiprocess import multi_process
from config import CnewsConfig


def funcs(x, y, z=False):
    return x+y

if __name__ == '__main__':
    # with Pool(6) as pool:
    #     pool.map(func, tqdm(list(zip(*[range(10000000), range(10000000)]))))
    multi_process(funcs, tqdm(list(zip(*[range(10000000), range(10000000)]))), 6, is_queue=False)
    # for data in tqdm(list(zip(*[range(10000000), range(10000000)]))):
    #     func(data)