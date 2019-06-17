import tensorflow as tf
import numpy
import matplotlib.pyplot as mPlot
from tensorflow.examples.tutorials.mnist import input_data

# 表示输入和输出的维数
n_input = 784
n_output = 10

# 按照正态分布设置权重参数, 一下分别表示 卷积层参数 和 全连接层参数
weights = {
    'wc1': tf.Variable(tf.random_normal(shape=[3, 3, 1, 64], stddev=0.1)),  # 分别指定过滤器的h, w, 当前过滤器连接的输入的深度(灰度图深度为1), 最终要得到的特征图的数量(也就是本层的输出的深度 )
    'wc2': tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.1)),

    'wd1': tf.Variable(tf.random_normal(shape=[7*7*128, 1024], stddev=0.1)),  # 暂时没有明白这个什么意思，需要看一下前面的视频
    'wd2': tf.Variable(tf.random_normal(shape=[1024, n_output], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal(shape=[64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal(shape=[128], stddev=0.1)),

    'bd1': tf.Variable(tf.random_normal(shape=[1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal(shape=[n_output], stddev=0.1))
}
