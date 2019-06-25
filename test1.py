# import tensorflow as tf
# import numpy
# # import pandas
# import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
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


####
# mnist = input_data.read_data_sets('data/', one_hot=True)
# print('type of mnist is %s' % (type(mnist)))
# print('number of train data is %d ' % (mnist.train.num_examples))
# print('number of test data is %d ' % (mnist.test.num_examples))

# trainImg = mnist.train.images
# trainLabel = mnist.train.labels
# testImg = mnist.test.images
# testLabel = mnist.test.labels
#
# print('MNIST loaded')

# print('type of trainImg is %s' % type(trainImg))
# print('type of trainLabel is %s' % type(trainLabel))
# print('type of testImg is %s' % type(testImg))
# print('type of testLabel is %s' % type(testLabel))
#
# print('shape of trainImg is %s' % (trainImg.shape,))
# print('shape of trainLabel is %s' % (trainLabel.shape,))
# print('shape of testImg is %s' % (testImg.shape,))
# print('shape of testLabel is %s' % (testLabel.shape,))
#
# print('#######################')
# print()
# nsample = 5
# randidx = numpy.random.randint(trainImg.shape[0], size=nsample)  # 返回一个包含5个随机选取的样例行下标的列表
# print('shape[0]: ', trainImg.shape[0])
# print('randidx: ', randidx)
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

# x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
# y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# W = tf.Variable(tf.zeros(shape=[784, 10]))
# b = tf.Variable(tf.zeros(shape=[10]))
#
# actv = tf.nn.softmax(tf.matmul(x, W) + b)
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
#
# learing_rate = 0.01
# optm = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(cost)
#
# prediction = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))  # 1表示行，查看actv返回的下标和y返回的下标是否一致
# accr = tf.reduce_mean(tf.cast(prediction, 'float'))  # 计算精确度
# init = tf.global_variables_initializer()
#
# # sess = tf.InteractiveSession()
# train_epochs = 50  # 训练轮数
# batch_size = 100  # 每次迭代选择的样本数
# display_step = 5  # 每5轮显示一次
#
# sess = tf.Session()
# sess.run(init)
#
# for epoch in range(train_epochs):
#     avg_cost = 0.0
#     num_batch = int(mnist.train.num_examples/batch_size)
#
#     for i in range(num_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         feeds = {x: batch_xs, y: batch_ys}
#         sess.run(optm, feed_dict=feeds)
#         avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
#
#     #display
#     if epoch%display_step == 0:
#         feeds_train = {x: batch_xs, y: batch_ys}
#         feeds_test = {x: mnist.test.images, y: mnist.test.labels}
#         train_acc = sess.run(accr, feed_dict=feeds_train)
#         test_acc = sess.run(accr, feed_dict=feeds_test)
#         print('Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f'
#               % (epoch, train_epochs, avg_cost, train_acc, test_acc))
#
# sess.close()
# print('DONE')

####
####



# for x in range(5):
#     name = 'wangpchao'
#     print(name)
#
# print('循环结束')
# print(name)



# excelFile = pandas.read_excel('C://Users//just_thinking//Desktop//四级练习得分统计.xlsx')
# excelFile = pandas.read_excel('data/excel1.xlsx',sheet_name='Sheet1')
# data = excelFile.values
# data = excelFile.head()
# data = excelFile.ix[[0, 4], ['晚饭（9）']].values
# data = excelFile.ix[[0, 4], [0, 3]].values

# print(format(data))
# print('获取到的值：\n{0}'.format(data))
#
# lineNumbers = excelFile.index.values
# print(len(lineNumbers))
#
# print(excelFile.columns.values)
#
# testData = []
# for i in excelFile.index.values:
#     rowData = excelFile.ix[i, excelFile.columns.values].to_dict()
#     testData.append(rowData)
# print('最终得到的数据：{0}'.format(testData))

# food_info = pandas.read_csv('data/food_info.csv')

# print(food_info.columns)
# print(food_info.shape)
# print(food_info.loc[0])


# # 2019年6月19日21:40:09
# # 输出以g为单位的列
# import pandas
# food_info = pandas.read_csv('data/food_info.csv')
# endWithG = []
# food_info_columns = food_info.columns.tolist()
# for item in food_info_columns:
#     if item.endswith('(g)'):
#         endWithG.append(item)
#
# gramdf = food_info[endWithG]
# print(gramdf.head())


# # 2019年6月19日21:39:56
# # 添加一列，并赋值
# import pandas
# food_info = pandas.read_csv('data/food_info.csv')
# Iron_grams = food_info['Iron_(mg)']/1000
# food_info['Iron_(g)'] = Iron_grams
# print(food_info[['Iron_(mg)', 'Iron_(g)']])


# # 2019年6月19日21:40:27
# # 排序
# import pandas
# food_info = pandas.read_csv('data/food_info.csv')
# food_info.sort_values('Water_(g)', inplace=True, ascending=False)  # 默认是升序排序,使用ascending=False之后变为降序
# print(food_info['Water_(g)'])


# # 2019年6月19日21:38:22
# # 使用 pandas.isnull 和 pandas.notnull 表示是否为缺失值
# import pandas
# titanic_survival = pandas.read_csv('data/titanic_train.csv')
# age = titanic_survival['Age']
#
# ageIsNull = pandas.isnull(age)  # True 缺失值 false表示有值
# ageNotNull = pandas.notnull(age)    # True 表示有值 False 表示缺失值
# # print(ageIsNull)
# ageFalse = age[ageIsNull]
# # print(ageFalse)


# 2019年6月19日21:38:02
# 计算某一列的平均值，如果这一列中含有缺失值，那么计算结果为nan，需要提前处理
# import pandas
# titanic_survival = pandas.read_csv('data/titanic_train.csv')
# meanAge = sum(titanic_survival['Age'])/len(titanic_survival['Age'])
# print(meanAge)


# # 2019年6月19日21:44:23
# # 使用剔除缺失值的方式计算均值
# import pandas
# titanic_survival = pandas.read_csv('data/titanic_train.csv')
#
# age = titanic_survival['Age']
# ageNotNull = pandas.notnull(age)  # 如果不是缺失值，则会返回 True
#
# # goodAges = titanic_survival['Age'][ageNotNull == True]
# goodAges = age[ageNotNull]
#
# # print(goodAges)
# # print(type(goodAges))
# meanAge = sum(goodAges)/len(goodAges)
# print(meanAge)
#
# # # 视频中计算均值的方式
# # ageIsNull = pandas.isnull(age)
# # goodAge =  titanic_survival['Age'][ageIsNull == False]
# # # print(goodAge)  # 此时输出的都是非缺失值
# # meanAge = sum(goodAge)/len(goodAge)
# # print(meanAge)


# # 2019年6月20日09:24:33
# # 直接调用函数求均值
# import pandas
# titanic_survival = pandas.read_csv('data/titanic_train.csv')
# meanAge = titanic_survival['Age'].mean()    # 注意调用函数的时候，不带括号表示函数赋值，将函数功能赋值给新的变量；带括号表示将函数的结果赋值给变量
# # meanAge = titanic_survival['Age'].mean
# # meanAge()
# print(meanAge)


# 2019年6月20日09:54:43
# 按照仓位等级进行统计
# passengerClasses = [1, 2, 3]
# faresByClass = {}
# for this_class in passengerClasses:
#     pass


# 2019年6月21日10:55:05
# 学习使用sklearn进行svm机器学习
# import numpy
# from sklearn import datasets
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC
#
# iris = datasets.load_iris()
# X = iris['data'][:, (2,3)]  # 花瓣长度，花瓣宽度
# y = (iris['target'] == 2).astype(numpy.float64)
#
# svm_clf = Pipeline((
#     ('scaler', StandardScaler()),
#     ('linear_svc', LinearSVC(C=1, loss='hinge')),
# ))
#
# svm_clf.fit(X, y)
# print(svm_clf.predict([[1.0, 1]]))


# 2019年6月21日10:59:14
'''
学习pipeline
 可以把多个评估器链接成一个。这个是很有用的，因为处理数据的步骤一般都是固定的，例如特征选择、标准化和分类。Pipeline 主要有两个目的:

    便捷性和封装性 你只要对数据调用 fit和 predict 一次来适配所有的一系列评估器。
    联合的参数选择 你可以一次grid search管道中所有评估器的参数。
    安全性 训练转换器和预测器使用的是相同样本，管道有助于防止来自测试数据的统计数据泄露到交叉验证的训练模型中。

管道中的所有评估器，除了最后一个评估器，管道的所有评估器必须是转换器。 (例如，必须有 transform 方法). 最后一个评估器的类型不限（转换器、分类器等等）
'''

# # 2019年6月21日14:12:32
# # 构建管道
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# estimators = [('reduce_dim', PCA()), ('clf', SVC())]
# pipe = Pipeline(estimators)
# print(pipe)
# print(pipe.steps[0])  # 按照索引输出 steps 中的内容
# print('===')
# print(pipe[0])  # 按照索引输出管道中评估器的内容
# print('===')
# print(pipe['clf'])  # 按照索引输出内容
# print('===')
# print(pipe.steps)
#
# # 管道中的评估器参数可以通过 <estimators>__<parameter> = newValue 的语义来访问，例如下面的 clf__c
# pipe.set_params( clf__C = 10)
# print(pipe)


# # 2019年6月21日14:12:13
# # 使用 make_pipeline 构建管道，它接受过个评估器并返回一个管道，自动填充评估器的名称
# from sklearn.pipeline import make_pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import Binarizer
#
# myPipe =  make_pipeline(Binarizer(), MultinomialNB())
# print(myPipe)


# # 2019年6月21日14:52:00
# # 网格搜索
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
#
# estimators = [('reduce_dim', PCA()), ('clf', SVC())]
# pipe = Pipeline(estimators)
#
# param_grid = dict(reduce_dim__n_components = [2, 5, 10],
#                   clf__C = [0.1, 10, 100])
# grid_search = GridSearchCV(pipe, param_grid = param_grid)
# print(grid_search)


# # 2019年6月21日16:09:56
# # 非线性svm分类
# from sklearn.datasets import make_moons
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC
# from sklearn.preprocessing import PolynomialFeatures
#
# estimators = [
#     ('pol_feature', PolynomialFeatures(degree=3)),
#     ('scaler', StandardScaler()),
#     ('svm_clf', LinearSVC(C=10, loss='hinge'))]
#
# polynomial_svm_clf = Pipeline(estimators)
# polynomial_svm_clf.fix()

# # 2019年6月21日21:56:55
# # 使用shuffle函数打乱矩阵中的数据（多维数组按照 行 打乱）
# import numpy
# array1 = numpy.array([[1,2,3],
#                       [4,5,6],
#                       [7,8,9],
#                       [12,13,14]])
# numpy.random.shuffle(array1)
# print(array1)


# # 2019年6月22日15:13:30
# # 使用numpy读取数据
# import numpy
# file = 'data/train_1.txt'
# cubFile = numpy.loadtxt(fname=file, delimiter=',')
# print(cubFile.shape)
# print(type(cubFile[0][1]))
# print(cubFile[0][1] + 2)


# # 2019年6月24日09:40:39
# import sklearn.metrics
# print(help(sklearn.metrics))

# 2019年6月25日17:26:55
l1 = ('hello', 'wpc')
def say(a, b):
     print(a,b)

# apply(say, l1)
l1.apply(say)
