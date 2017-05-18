class Config(object):
    def __init__(self):
        # self.input_len = 784
        # self.hidden_num = 200

        self.struct = [None, 500, 100] # input_len=none, h1 = 500, h2 = 100

        ## graph data
        self.file_path = "./GraphData/cora/graph.txt"
        self.label_file_path = "GraphData/cora/group.txt"

        ## embedding data
        self.embedding_filename = "citeseer"

        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 5
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 15
        
        ## para for training
        self.batch_size = 1024
        self.epochs_limit = 5000
        self.learning_rate = 0.001

        
        
