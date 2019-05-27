import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# import input_data

# a = 3
#
# w = tf.Variable([[0.5, 1.0]])
# x = tf.Variable([[2.0], [1.0]])
#
# y = tf.matmul(w, x)
#
# print(w)
#
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(y.eval())
#     # print(sess.run(y))



# state = tf.Variable(0)
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     for i in range(3):
#         sess.run(update)
#         # print(state.eval())
#         print(sess.run(state))
#
#         savePath = saver.save(sess=sess, save_path='C://Users//justThinking//Desktop//tempSess')
#         print('save in: ', savePath)

# input1 = tf.placeholder(dtype=tf.float32, shape=(1, 1))
# input2 = tf.placeholder(dtype=tf.float32, shape=(1, 1))
# output = tf.matmul(input1, input2)
#
# with tf.Session() as sess:
#     print(sess.run([output], feed_dict={input1: [[7]], input2: [[2]]}))


# num_points = 1000
# vectors_set = []
# for i in range(num_points):
#     x1 = numpy.random.normal(0.0, 0.55)
#     y1 = x1*0.1 + 0.3 + numpy.random.normal(0.0, 0.03)
#     vectors_set.append([x1, y1])
#
# x_data = [v[0] for v in vectors_set]
# y_data = [v[1] for v in vectors_set]
#
#
#
#
# W = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0, dtype=tf.float32, name='W'))
# b = tf.Variable(tf.zeros(shape=[1]), name='b')
# y = W * x_data + b
#
# # 定义损失函数
# loss = tf.reduce_mean(tf.square(y - y_data))
#
# # 选择梯度下降优化器
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
#
# # 定义训练过程
# train = optimizer.minimize(loss, name='train')
#
# init_op = tf.global_variables_initializer()
#
# finalW = 0.0
# finalb = 0.0
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print('W = ', sess.run(W), 'b = ', sess.run(b))
#
#     for step in range(20):
#         sess.run(train)
#         print('W = ', sess.run(W), ' b = ', sess.run(b), ' loss = ', sess.run(loss))
#         finalW = sess.run(W)
#         finalb = sess.run(b)
#
# plt.scatter(x_data, y_data, c='r')
# plt.plot(x_data, finalW * x_data + finalb)
# plt.show()
#
# print('finalW: ', finalW, 'finalb: ', finalb)


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

prediction = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
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
        print('Epoch: %03d/%03d')



sess.close()




