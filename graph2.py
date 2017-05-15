import numpy as np
class Graph(object):
    def __init__(self, file_path):
        self.st = 0
        self.is_epoch_end = False
        fin = open(file_path, "r")
        firstLine = fin.readline().strip().split(" ")
        self.N = int(firstLine[0])
        self.E = int(firstLine[1])
        self.__is_epoch_end = False
        self.adj_matrix = np.zeros([self.N, self.N], np.int_)
        self.__links = np.zeros([self.E + ng_sample_ratio*self.N,3], np.int_)
        count = 0
        for line in fin.readlines():
            line = line.strip().split(' ')
            self.adj_matrix[int(line[0]),int(line[1])] += 1
            self.adj_matrix[int(line[1]),int(line[0])] += 1
            self.__links[count][0] = int(line[0])
            self.__links[count][1] = int(line[1])
            self.__links[count][2] = 1
            count += 1
        fin.close()
        if (ng_sample_ratio > 0):
            self.__negativeSample(ng_sample_ratio * self.N, count, edges.copy())
        self.__order = np.arange(self.N)
        print("getData done")
        print("Vertexes : %d  Edges : %d ngSampleRatio: %f" % (self.N, self.E, ng_sample_ratio))
