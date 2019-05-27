import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
# print('type of mnist is %s' % (type(mnist)))
# print('number of train data is %d ' % (mnist.train.num_examples))
# print('number of test data is %d ' % (mnist.test.num_examples))

trainImg = mnist.train.images
trainLabel = mnist.train.labels
testImg = mnist.test.images
testLabel = mnist.test.labels

print('MNIST loaded')

# print('type of trainImg is %s' % type(trainImg))
# print('type of trainLabel is %s' % type(trainLabel))
# print('type of testImg is %s' % type(testImg))
# print('type of testLabel is %s' % type(testLabel))
#
print('shape of trainImg is %s' % (trainImg.shape,))
print('shape of trainLabel is %s' % (trainLabel.shape,))
print('shape of testImg is %s' % (testImg.shape,))
print('shape of testLabel is %s' % (testLabel.shape,))

print('#######################')
print()
nsample = 5
randidx = numpy.random.randint(trainImg.shape[0], size=nsample)  # 返回一个包含5个随机选取的样例行下标的列表
# print('shape[0]: ', trainImg.shape[0])
print('randidx: ', randidx)
# print(trainLabel[0])


# 显示随机选取的5个样本
# for i in randidx:
#     curr_img = numpy.reshape(trainImg[i, :], (28, 28))
#     curr_label = numpy.argmax(trainLabel[i, :])
#     print('curr_label: ', curr_label)
#     print('trainLabel[%d]: ' % (i), trainLabel[i])
#     print('' + str(i) + 'the training date label is ' + str(curr_label))
#
#     plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
#     plt.title('' + str(i) + 'the training data label is ' + str(curr_label))
#     plt.show()

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

actv = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))

learing_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(cost)

prediction = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))  # 1表示行，查看actv返回的下标和y返回的下标是否一致
accr = tf.reduce_mean(tf.cast(prediction, 'float'))  # 计算精确度
init = tf.global_variables_initializer()

# sess = tf.InteractiveSession()
train_epochs = 50  # 训练轮数
batch_size = 100  # 每次迭代选择的样本数
display_step = 5  # 每5轮显示一次

sess = tf.Session()
sess.run(init)

for epoch in range(train_epochs):
    avg_cost = 0.0
    num_batch = int(mnist.train.num_examples/batch_size)

    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch

    #display
    if epoch%display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print('Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f'
              % (epoch, train_epochs, avg_cost, train_acc, test_acc))

sess.close()
print('DONE')


