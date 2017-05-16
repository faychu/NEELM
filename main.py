#!/usr/bin/python2
# -*- coding: utf-8 -*-

from config import Config
from graph1 import Graph
from ELM_AE_o import ELM_AE
import time
import scipy.io as sio
import numpy as np
import tensorflow as tf
from utils.utils import *


if __name__ == "__main__":
    config = Config()
    graph_data = Graph(config.file_path)
    config.struct[0] = graph_data.N
    sess = tf.Session()

    print(graph_data.adj_matrix)

    model = ELM_AE(sess,config)
    model.do_variables_init()

    last_loss = np.inf
    converge_count = 0
    time_consumed = 0
    batch_n = 0
    epochs = 0
    while(True):
        mini_batch= graph_data.sample(config.batch_size)
        st_time = time.time()
        model.fit(mini_batch)
        batch_n+=1
        time_consumed += time.time()-st_time
        print('Mini-batch: %d  fit time: %.2f' % (batch_n, time_consumed))
        if graph_data.is_epoch_end:
            epochs+=1
            loss = 0
            embedding = None
            while(True):
                mini_batch = graph_data.sample(config.batch_size,do_shuffle=False)
                loss += model.get_loss(mini_batch)
                if embedding is None:
                    embedding= model.get_embedding(mini_batch)
                else:
                    embedding= np.vstack((embedding,model.get_embedding(mini_batch)))

                if graph_data.is_epoch_end:
                    break

            print("Epoch: %d Loss: %.3f, Train time_consumed: %.3fs" % (epochs,loss,time_consumed))
            if epochs % 10 ==0:
                check_link_reconstruction(embedding,graph_data,[1000,3000,5000,7000,9000,10000])

            if (loss > last_loss+3000):
                converge_count += 1
                if converge_count > 10:
                    print("model converge terminating")
                    # check_link_reconstruction(embedding, graph_data, [1000,3000,5000,7000,9000,10000])
                    break
            if epochs > config.epochs_limit:
                print("exceed epochs limit terminating")
                break
            last_loss = loss



