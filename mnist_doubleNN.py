import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(train_dir='data/', one_hot=True)

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# 输入输出
x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])  # ?*784
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])  # ?*10

# 神经网络的参数
# stddev = 0.1
# 初始化权重参数
weights = {
    'w1': tf.Variable(tf.random_normal(shape=[n_input, n_hidden_1], stddev=0.1)),  # 784*256
    'w2': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),  # 265*128
    'out': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_classes], stddev=0.1))  # 128*10
}
# 初始化b参数
biases = {
    # 'b1': tf.Variable(tf.random_normal(shape=[n_hidden_1], stddev=0.1)),
    # 'b2': tf.Variable(tf.random_normal(shape=[n_hidden_2], stddev=0.1)),
    # 'out': tf.Variable(tf.random_normal(shape=[n_classes]))
    'b1': tf.Variable(tf.zeros(shape=[n_hidden_1])),
    'b2': tf.Variable(tf.zeros(shape=[n_hidden_2])),
    'out': tf.Variable(tf.zeros(shape=[n_classes]))
}
print('network readly\n')

# 多层感知机
def multilayer_perceptron(X, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['w1']), biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    return tf.matmul(layer_2, weights['out']) + biases['out']


# 使用多层感知机得到预测结果
prediction = multilayer_perceptron(X=x, weights=weights, biases=biases)

# 损失和优化
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))  # 已经弃用的方法
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))  # 计算logits个labels之间的softmax交叉熵
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
is_equal = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))  # 预测值和真实值是否相等
accuracy = tf.reduce_mean(tf.cast(is_equal, dtype=tf.float32))  # 使用均值衡量准确率

init = tf.global_variables_initializer()
print('functions ready')

train_epochs = 20
batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)
display_step = 4

sess = tf.Session()
sess.run(init)

for epoch in range(train_epochs):
    avg_loss = 0.0
    for i in range(total_batch):
        total_loss = 0.0
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 取出当前 批次(batch) 中的x和y
        feeds = {x: batch_xs, y: batch_ys}  # 提前准备好要喂入的x和y
        sess.run(optimizer, feed_dict=feeds)
        total_loss += sess.run(loss, feed_dict=feeds)

    avg_loss = total_loss/total_batch

    if (epoch + 1) % display_step == 0:
        test_feeds = {x: mnist.test.images, y: mnist.test.labels}  # 准备数据，用于测试
        print('Epoch: %03d/%03d  loss:%.9f' % (epoch, train_epochs, avg_loss))

        train_accuracy = sess.run(accuracy, feed_dict=feeds)
        print('Trian accuracy: %.3f' % (train_accuracy))
        test_accuracy = sess.run(accuracy, feed_dict=test_feeds)  # 计算方式和 训练精确度 相同，区别只是输入的数据不同
        print('Test accuracy: %.3f' % (test_accuracy))

print('Optimization finished')



