#!/usr/bin/python2
# -*- coding: utf-8 -*-

from config import Config
from graph1 import Graph
import time
import scipy.io as sio
import numpy as np

if __name__ == "__main__":
    config = Config()
    graph_data = Graph(config.file_path)
    config.struct[0] = graph_data.N

    print(graph_data.adj_matrix)

    last_loss = np.inf
    converge_count = 0
    time_consumed = 0
    batch_n = 0
    epochs = 0
    while(True):
        mini_batch= graph_data.sample(config.batch_size)
        st_time = time.time()

