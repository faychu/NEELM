import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from config import Config

def orthonormal(size):
    assert size[0] > size[1]
    a = np.random.randn(size[0],size[1])
    q,r = np.linalg.qr(a,mode='reduced')
    return q

omega = 1.

class ELM_AE(object):
    def __init__(self, sess, config):
        '''
        Args:
              sess : TensorFlow session.
              batch_size : The batch size (N)
              input_len : The length of input. (L)
              hidden_num : The number of hidden node. (K)

        '''

        self._sess = sess
        self.struct = config.struct
        struct = self.struct
        self._batch_size = config.batch_size


        ############ define variables ##################
        self._W = tf.Variable(
            orthonormal([struct[1], struct[2]]),  # (LxK)
            trainable=False, dtype=tf.float32, name='W')
        self._b = tf.Variable(
            orthonormal([self.struct[2], 1]),
            trainable=False, dtype=tf.float32, name='b')
        self._beta_s = tf.Variable(
            tf.zeros([struct[2], struct[1]]),
            trainable=False, dtype=tf.float32)
        # self.W_i = tf.Variable(tf.random_normal([struct[0], struct[1]]), name='W_i', trainable=True)
        # self.b_i = tf.Variable(tf.zeros([struct[1]]), name='b_i', trainable=True)
        self.W_o = tf.Variable(tf.random_normal([struct[1], struct[0]]), name='W_o', trainable=True)
        self.b_o = tf.Variable(tf.zeros([struct[0]]), name='b_o', trainable=True)
        ###############################################

        ############## define input ###################
        self.adjacent_matriX = tf.placeholder("float", [None, None])
        self.X = tf.placeholder("float", [None, config.struct[0]])
        ###############################################
        self.__make_compute_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)

    def __make_compute_graph(self):
        struct = self.struct
        self.X0 = tf.nn.sigmoid(tf.matmul(self.X, self.W_i) + self.b_i)
        self.H0 = tf.nn.sigmoid(tf.matmul(self.X0, self._W) + tf.reshape(self._b, [struct[2]], name='H0'))  # (NxK)
        self.H0_T = tf.transpose(self.H0, name='H0_T')  # (KxN)
        #  _beta_s = (H_T*H + I/om)^(-1)*H_T*T  (KxL)
        identity = tf.constant(np.identity(struct[2]), dtype=tf.float32)
        self._beta_s = tf.matmul(
            tf.matmul(tf.matrix_inverse(tf.matmul(self.H0_T, self.H0) + identity / omega), self.H0_T), self.X0)
        # self.embedding = tf.nn.sigmoid(tf.matmul(self.X0, tf.transpose(self._beta_s)))
        # self.embedding = self.H0
        self.newX0 = tf.matmul(self.H0, self._beta_s)
        self.newX = tf.nn.sigmoid(tf.matmul(self.newX0, self.W_o)+self.b_o)

    def __make_loss(self,config):
        def get_1st_loss(H, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
            L = D - adj_mini_batch  ## L is laplation-matriX
            return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X) * B, 2))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases])
            return ret

        # Loss function
        self.loss_2nd = get_2nd_loss(self.X, self.newX, config.beta)
        self.loss_1st = get_1st_loss(self.H0, self.adjacent_matriX)
        self.loss_reg = get_reg_loss([self.W_o,self.W_i], [self.b_o,self.b_i])
        return config.gamma * self.loss_1st + config.alpha * self.loss_2nd + config.reg * self.loss_reg

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self._sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self._sess, path)
        self.is_Init = True

    def do_variables_init(self):
        init = tf.global_variables_initializer()
        self._sess.run(init)
        self.is_Init = True

    def __get_feed_dict(self, data):
        return {self.X: data.X, self.adjacent_matriX: data.adjacent_matriX}

    def fit(self, data):
        if (not self.is_Init):
            print("Warning: the model isn't initialized, and will be initialized randomly")
            self.do_variables_init()
        feed_dict = self.__get_feed_dict(data)
        _ = self._sess.run(self.optimizer, feed_dict=feed_dict)

    def get_loss(self, data):
        if (not self.is_Init):
            print("Warning: the model isn't initialized, and will be initialized randomly")
            self.do_variables_init()
        feed_dict = self.__get_feed_dict(data)
        return self._sess.run(self.loss, feed_dict=feed_dict)


    def get_accuracy(self,x,y):
        if not self._init: self.init()
        return self._sess.run(self.accuracy, {self._x0: x, self.y:y})

    def get_W(self,x,y):
        return self._sess.run(self.W, {self._x0: x, self.y: y})

    def get_embedding(self,data):
        return self._sess.run(self.H0, feed_dict=self.__get_feed_dict(data))

    def close(self):
        self._sess.close()


