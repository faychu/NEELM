class Config(object):
    def __init__(self):
        # self.input_len = 784
        # self.hidden_num = 200
        self.batch_size = 300
        self.struct = [None, 500, 100] # input_len=none, h1 = 500, h2 = 100

        ## graph data
        self.file_path = "./GraphData/ca-Grqc.txt"
        ## embedding data
        self.embedding_filename = "ca-Grac" 

        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 1
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 15
        
        ## para for training
        self.batch_size = 1024
        self.epochs_limit = 1000
        self.learning_rate = 0.001

        
        
