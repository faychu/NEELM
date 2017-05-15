from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)   # 读取数据集
# 建立抽象模型
x = tf.placeholder(tf.float32, [None, 784]) # 占位符
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
a = tf.nn.softmax(tf.matmul(x, W) + b)
# 定义损失函数和训练方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))  # 损失函数为交叉熵
optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降法，学习速率为0.5
train = optimizer.minimize(cross_entropy) # 训练目标：最小化损失函数
# Test trained model
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train.run({x: batch_xs, y: batch_ys})
        print(sess.run(accuracy,{x: batch_xs, y: batch_ys}))
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))