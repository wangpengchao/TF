import numpy
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainImg = mnist.train.images
trainLabel = mnist.train.labels
testImg = mnist.test.images
testLabel = mnist.test.labels
print('MNIST loaded\n')


# 输出mnist数据集的基本信息
print('type of mnist is %s' % (type(mnist)))
print('number of train data is %d ' % (mnist.train.num_examples))
print('number of test data is %d ' % (mnist.test.num_examples))
print()

print('type of trainImg is %s' % type(trainImg))
print('type of trainLabel is %s' % type(trainLabel))
print('type of testImg is %s' % type(testImg))
print('type of testLabel is %s' % type(testLabel))
print()

print('shape of trainImg is %s' % (trainImg.shape,))
print('shape of trainLabel is %s' % (trainLabel.shape,))
print('shape of testImg is %s' % (testImg.shape,))
print('shape of testLabel is %s' % (testLabel.shape,))
print()

nsample = 5
randidx = numpy.random.randint(trainImg.shape[0], size=nsample)  # 返回一个包含5个随机选取的样例行下标的列表
# print('shape[0]: ', trainImg.shape[0])
# print('randidx: ', randidx)
# print(trainLabel[0])

# 显示随机选取的5个样本
for i in randidx:
    current_img = numpy.reshape(trainImg[i, :], (28, 28))
    current_label = numpy.argmax(trainLabel[i, :])
    print('curr_label: ', current_label)
    print('trainLabel[%d]: ' % (i), trainLabel[i])
    print('' + str(i) + 'the training date label is ' + str(current_label))

    plt.matshow(current_img, cmap=plt.get_cmap('gray'))
    plt.title('' + str(i) + 'the training data label is ' + str(current_label))
    plt.show()
