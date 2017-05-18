import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import random

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)
    
def check_link_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        # norm = np.sqrt(np.sum(embedding*embedding, axis=1)).reshape([-1,1])
        # norm_embedding = embedding/norm
        # similarity = getSimilarity(norm_embedding).reshape(-1)
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            if (x == y):
                continue
            count += 1
            if (data.adj_matrix[x][y] == 1):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))


    #############################################
def check_classification(X, Y, test_ratio):
    macro_f1 = []
    micro_f1 = []
    accuracy = []
    auc_score = []
    for ratio in test_ratio:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = ratio)
        clf = svm.LinearSVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        macro_f1.append(f1_score(y_test, y_pred, average = "macro"))
        micro_f1.append(f1_score(y_test, y_pred, average = "micro"))
        accuracy.append(accuracy_score(y_test,y_pred))
        # auc_score.append(roc_auc_score(y_test,y_pred))
    for ratio, score in zip(test_ratio, macro_f1):
        print("macro_f1: test ratio %.1f : %.2f" % (ratio, score))
    for ratio, score in zip(test_ratio, micro_f1):
        print("micro_f1: test ratio %.1f : %.2f" % (ratio, score))
    for ratio, score in zip(test_ratio, accuracy):
        print("accuracy: test ratio %.1f : %.2f" % (ratio, score))
    #############################################
