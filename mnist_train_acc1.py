import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 指定mnist数据集的位置，
mnist = input_data.read_data_sets('data/', one_hot=True)

trainImg = mnist.train.images
trainLabel = mnist.train.labels
testImg = mnist.test.images
testLabel = mnist.test.labels
print('MNIST loaded')

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])  # ?*784 大小的二维矩阵
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # ?*10 大小的二维矩阵
W = tf.Variable(tf.zeros(shape=[784, 10]))  # W参数为 784*10 大小的二维矩阵
b = tf.Variable(tf.zeros(shape=[10]))  # b参数为 长度为 10 的向量

actv = tf.nn.softmax(tf.matmul(x, W) + b)  # 预测结果: Tensor("Softmax:0", shape=(?, 10), dtype=float32)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))  # y*log(actv)获取真正值得交叉熵损失

# print('y.shape: ', y.shape)  # y.shape:  (?, 10)
# print('actv.shape: ', actv.shape)  # actv.shape:  (?, 10)

learn_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)  # 梯度下降求解的优化器

# 1表示按行获取最大值得下标，查看查看预测结果actv返回的下标和真实的y的下标是否一致
prediction = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(prediction, 'float'))  # 使用均值衡量准确率，将true和false转换为float类型（true-》1.0， false-》0.0）
init = tf.global_variables_initializer()

# sess = tf.InteractiveSession()
train_epochs = 50  # 训练轮数
batch_size = 100  # 每100个为一个batch
batchs_amount = int(mnist.train.num_examples/batch_size)  # 表示每轮训练中有 batchs_amount 个 batch
display_step = 5  # 每5轮显示一次

sess = tf.Session()
sess.run(init)

# 总共训练 train_epochs 轮，每一轮中分为 batch_amount 个训练批次， 每个批次中有 batch_size 个数据
for epoch in range(train_epochs):
    avg_cost = 0.0
    for i in range(batchs_amount):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optimizer, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)/batchs_amount

    # print('batch_xs: ', batch_xs)
    # print('batch_ys: ', batch_ys)
    # print()

    #每5轮显示一次
    if epoch%display_step == 0:

        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}

        train_acc = sess.run(accuracy, feed_dict=feeds_train)
        test_acc = sess.run(accuracy, feed_dict=feeds_test)

        print('Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f'
              % (epoch, train_epochs, avg_cost, train_acc, test_acc))

sess.close()
print('DONE')


