import numpy

train0 = numpy.loadtxt(fname='data/train_0.txt', delimiter=',')
train1 = numpy.loadtxt(fname='data/train_1.txt', delimiter=',')

numpy.random.shuffle(train0)
newTrain0 = train0[0:300]

train = numpy.vstack((newTrain0, train1))  # 按照纵轴堆叠成为新的矩阵

numpy.random.shuffle(train)  # 将数据打乱

trainLabel = train[:400, 0].astype(numpy.int)
trainSet = train[0:400, 1:]  # 400条数据作为 训练集

testLabel = train[400:, 0].astype(numpy.int)
testSet = train[400:, 1:]  # 165条数据作为 测试集

# 数据集进行归一化操作

