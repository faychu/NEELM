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
        self._batch_size = config.batch_size
        self._input_len = config.input_len
        self._hidden_num = config.hidden_num

        #  for train
        self._x0 = tf.placeholder(tf.float32, [None, self._input_len],name='input_x')  # (NxL）

        self._W = tf.Variable(
            orthonormal([self._input_len, self._hidden_num]),  # (LxK)
            trainable=False, dtype=tf.float32,name='W')
        self._b = tf.Variable(
            orthonormal([self._hidden_num, 1]),
            trainable=False, dtype=tf.float32, name='b')
        self._beta_s = tf.Variable(
            tf.zeros([self._hidden_num, self._input_len]),
            trainable=False, dtype=tf.float32)

        self.H0 = tf.sigmoid(tf.matmul(self._x0, self._W) + tf.reshape(self._b,[self._hidden_num],name='H0'))  # (NxK)
        self.H0_T = tf.transpose(self.H0,name='H0_T')  # (KxN)

        #  _beta_s = (H_T*H + I/om)^(-1)*H_T*T  (KxL)
        identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
        self._beta_s = tf.matmul(
            tf.matmul(tf.matrix_inverse(tf.matmul(self.H0_T, self.H0) + identity / omega), self.H0_T),self._x0)
        self.embedding = tf.sigmoid(tf.matmul(self._x0,tf.transpose(self._beta_s)))

        self.newx = tf.matmul(self.H0, self._beta_s)

        self.cost = tf.reduce_sum(tf.pow((self.newx - self._x0), 2))




        # for test
        self.y = tf.placeholder(tf.float32, [None, 10],name='y')
        self.W = tf.Variable(tf.zeros([self._hidden_num, 10]),name='W_',trainable=True)
        self.b = tf.Variable(tf.zeros([10]),name='b_',trainable=True)
        a = tf.nn.softmax(tf.matmul(self.embedding, self.W) + self.b,name='a')

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(a), reduction_indices=[1]),name='cross_entropy')  # 损失函数为交叉熵
        optimizer = tf.train.GradientDescentOptimizer(0.5)  # 梯度下降法，学习速率为0.5
        self.train = optimizer.minimize(cross_entropy,name='train')  # 训练目标：最小化损失函数

        correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

        self._init = False
        self._feed = False

        writer = tf.summary.FileWriter('log',self._sess.graph)
        writer.close()

    def feed(self,x):
        '''
            Args :
              x : input array (N x L)
            '''

        if not self._init: self.init()
        self._sess.run(self.embedding, {self._x0: x})
        self._feed = True


    def init(self):
        self._sess.run(tf.global_variables_initializer())
        self._init = True

    def make_loss(self, config):
        def get_1st_loss(embedding, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
            L = D - adj_mini_batch  ## L is laplation-matriX
            return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(embedding), L), H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X) * B, 2))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.itervalues()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
            return ret

    def get_accuracy(self,x,y):
        if not self._init: self.init()
        return self._sess.run(self.accuracy, {self._x0: x, self.y:y})

    def get_W(self,x,y):
        return self._sess.run(self.W, {self._x0: x, self.y: y})

    def get_embedding(self,x):
        return self._sess.run(self.embedding, {self._x0: x})

    def get_cost(self,x):
        return self._sess.run(self.cost,{self._x0: x})

    def get_beta(self,x):
        return self._sess.run(self._x0,{self._x0: x})

if __name__ == "__main__":
    sess = tf.Session()

    # Get data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Construct ELM
    config = Config()
    print("batch_size : {}".format(config.batch_size))
    print("hidden_num : {}".format(config.hidden_num))

    elm = ELM_AE(sess, config)
    elm.init()
    for i in range(1000):

        train_x, train_y = mnist.train.next_batch(config.batch_size)
        # print(train_x)
        # print(train_y)
        # print("mul:",elm.get_mul(train_x))
        elm.feed(train_x)
        # print("embedding",elm.get_embedding(train_x))
        print("accuracy",elm.get_accuracy(train_x, train_y))
        print("cost", elm.get_cost(train_x))
        # print("beta",elm.get_beta(train_x))
    test_x, test_y = mnist.test.next_batch(config.batch_size)
    print(elm.get_accuracy(test_x, test_y))
