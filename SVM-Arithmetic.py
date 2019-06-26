import numpy
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sklearn.model_selection


# 实现简单的加载文件的功能
def load_text_file(file_name, delimiter=','):
    '''
    加载预处理之后的txt格式文件
    :param fileName: 包含文件名的文件路径
    :param delimiter: 使用的分隔符
    :return: ndarray格式的矩阵
    '''
    fFileName = file_name
    fDelimiter = delimiter
    return numpy.loadtxt(fname=fFileName, delimiter=fDelimiter)


# 生成 样本集 自动判定是否样本是否不平衡，然后如果不平衡则采取 下采样 的方式避免不平衡
def get_samples(train0_file, train1_file):
    train0 = load_text_file(train0_file)
    train1 = load_text_file(train1_file)
    train0Size = train0.shape[0]
    train1Size = train1.shape[0]
    # numpy.random.seed(42)  # 将随机的结果固定

    # 处理数据的不平衡问题 以 10:1 为不平衡界限
    if (train0.shape[0]/train1.shape[0] > 10):  # 0数据多
        train0Size = train1Size
        numpy.random.shuffle(train0)
        newTrain0 = train0[0:train0Size]
        trainSamples = numpy.vstack((newTrain0, train1))  # 按照纵轴堆叠成为新的矩阵
    elif (train0.shape[0]/train1.shape[0] < 0.1): # 1数据多
        train1Size = train0Size
        numpy.random.shuffle(train1)
        newTrain1 = train1[0:train1Size]
        trainSamples = numpy.vstack((newTrain1, train0))
    numpy.random.shuffle(trainSamples)  # 将数据打乱
    return trainSamples


# 从数据集中分离出 训练集 和 测试集，并进行归一化处理
def split_train_test(train_samples, testRatio):
    train_set, test_set = sklearn.model_selection.train_test_split(train_samples, test_size=testRatio)

    train_label = train_set[:, 0].astype(numpy.int)
    train_set = train_set[:, 1:]

    test_label = test_set[:, 0].astype(numpy.int)
    test_set = test_set[:, 1:]

    train_set = normalize(train_set, axis=0)
    test_set = normalize(test_set, axis=0)

    return train_label, train_set, test_label, test_set


def svm_train(train_label, train_set, test_label, test_set):
    estimators = [
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=0.1, loss="hinge", random_state=42))
    ]
    pipe = Pipeline(estimators)
    pipe.fit(train_set, train_label)
    predictResult = pipe.predict(test_set)

    tp = fp = fn = tn = 0
    for i in range(len(test_label)):
        if (test_label[i] == 1) and (predictResult[i] == 1):
            tp = tp + 1
        if (test_label[i] == 0) and (predictResult[i] == 1):
            fp = fp + 1
        if (test_label[i] == 1) and (predictResult[i] == 0):
            fn = fn + 1
        if (test_label[i] == 0) and (predictResult[i] == 0):
            tn = tn + 1
    # print('1样本准确率：', tp / (tp + fp))
    # print('1样本召回率：', tp / (tp + fn))
    # print('0样本准确率：', tn / (tn + fn))
    # print('0样本召回率：', tn / (tn + fp))
    return tp / (tp + fp), tp / (tp + fn), tn / (tn + fn), tn / (tn + fp)


if __name__ == '__main__':
    trainSamples = get_samples(train0_file='data/train_0.txt', train1_file='data/train_1.txt')
    trainLabel, trainSet, testLabel, testSet = split_train_test(train_samples=trainSamples, testRatio=0.2)
    r1, r2, r3, r4 = svm_train(trainLabel, trainSet, testLabel, testSet)
    print(r1, r2, r3, r4)
