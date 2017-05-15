import numpy as np
from utils.utils import *
# for blog, flickr, youtube, ca-Grqc
class Graph(object):
    def __init__(self, file_path):
        self.st = 0
        self.is_epoch_end = False
        fin = open(file_path, "r")
        firstLine = fin.readline().strip().split(" ")
        self.N = int(firstLine[0])
        self.E = int(firstLine[1])
        self.adj_matrix = np.zeros([self.N, self.N], np.int_)
        for line in fin:
            line = line.strip().split(' ')
            self.adj_matrix[int(line[0]),int(line[1])] += 1
            self.adj_matrix[int(line[1]),int(line[0])] += 1
        fin.close()
        self.__order = np.arange(self.N)
        print("getData done")
        print("Vertexes : %d  Edges : %d " % (self.N, self.E))

    # def load_label_data(self, filename):
    #     self.label = np.zeros([self.N], np.int_)
    #     with open(filename, 'r') as fin:
    #         for line in fin:
    #             line = line.

    def sample(self, batch_size, do_shuffle=True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.__order)
            else:
                self.__order = np.sort(self.__order)
            self.st = 0
            self.is_epoch_end = False

        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = self.__order[self.st:en]
        mini_batch.X = self.adj_matrix[index]
        mini_batch.adjacent_matriX = self.adj_matrix[index][:, index]
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch