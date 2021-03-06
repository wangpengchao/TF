import numpy
import xlrd
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import sklearn.model_selection
from sklearn.externals import joblib
import re


# 预处理文件：将xlsx文件处理成txt格式的文件
def pre_data():
    excel_file = xlrd.open_workbook('final_model.xlsx')
    sheet=excel_file.sheet_by_index(0)
    nrows = sheet.nrows
    with open('train_1.txt', 'w', encoding='utf-8') as train_file_1:
        with open('train_0.txt', 'w', encoding='utf-8') as train_file_0:
            for row in range(nrows):
                if (row == 0):
                    continue
                row_data = sheet.row_values(row)
                K_arer = row_data[7]
                if (K_arer == ''):
                    continue
                His_arer = row_data[16]
                if (His_arer == ''):
                    continue

                n_arer = row_data[10]
                if (type(n_arer) == str):
                    if (row > 0):
                        n_arer = float(str(n_arer).strip("'"))
                        row_data[10] = n_arer

                row_data[8] = round(row_data[7] - row_data[8], 2)
                row_data[9] = round(row_data[9] - row_data[7], 2)

                row_data[11] = round(row_data[10] - row_data[11], 2)
                row_data[12] = round(row_data[12] - row_data[10], 2)

                row_data[14] = round(row_data[13] - row_data[14], 2)
                row_data[15] = round(row_data[15] - row_data[13], 2)

                row_data[17] = round(row_data[17] - abs(row_data[16] - row_data[7])/row_data[17]*100, 2)


                row_data = [row_data[0]] + row_data[7:18]
                str_data = str(row_data).strip('[]')
                print('第 ' + str(row) + '行：',str_data)
                if row_data[0] > 0.5:
                    train_file_1.write(str(str_data) + '\n')
                else:
                    train_file_0.write(str(str_data) + '\n')


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
def split_train_test(train_samples, test_ratio):
    train_set, test_set = sklearn.model_selection.train_test_split(train_samples, test_size=test_ratio)

    train_label = train_set[:, 0].astype(numpy.int)
    train_set = train_set[:, 1:]

    test_label = test_set[:, 0].astype(numpy.int)
    test_set = test_set[:, 1:]

    train_set = normalize(train_set, axis=0)
    test_set = normalize(test_set, axis=0)

    return train_label, train_set, test_label, test_set


# 进行训练前，先让用户决定是否自动获取最优超参数。得到的结果参数一并传到训练函数中
def get_best_parameter():
    pass


# 三种不同核函数的训练方法
# 1.线性核函数
def train_linear_kernel():
    estimators = [
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=0.1, loss="hinge", random_state=42))
    ]
    pipe = Pipeline(estimators)
    return pipe


# 2.poly核函数
def train_poly_kernel():
    estimators = [
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=2, coef0=1, C=0.1))
    ]
    pipe = Pipeline(estimators)
    return pipe


# 3.高斯核函数 rbf
def train_rbf_kernel():
    estimators = [
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.1))
    ]
    pipe = Pipeline(estimators)
    return pipe


# 保存训练好的模型
def save_model(pipe_model):
    joblib.dump(pipe_model, 'data/savedModel.m')
    print('保存成功。')


# 加载之前训练的模型
def load_model(file_name):
    joblib.load(filename=file_name)


def svm_train(kernel, train_label, train_set, test_label, test_set):

    # estimators = [
    #     ("poly_features", PolynomialFeatures(degree=2)),
    #     ("scaler", StandardScaler()),
    #     ("svm_clf", LinearSVC(C=0.1, loss="hinge", random_state=42))
    # ]
    # print('estimators', type(estimators))
    # pipe = Pipeline(estimators)

    if kernel == 'linear':
        pipe = train_linear_kernel()
    elif kernel == 'poly':
        pipe = train_poly_kernel()
    elif kernel == 'rbf':
        pipe = train_rbf_kernel()

    pipe.fit(train_set, train_label)
    predict_result = pipe.predict(test_set)

    tp = fp = fn = tn = 0
    for i in range(len(test_label)):
        if (test_label[i] == 1) and (predict_result[i] == 1):
            tp = tp + 1
        if (test_label[i] == 0) and (predict_result[i] == 1):
            fp = fp + 1
        if (test_label[i] == 1) and (predict_result[i] == 0):
            fn = fn + 1
        if (test_label[i] == 0) and (predict_result[i] == 0):
            tn = tn + 1
    # print('1样本准确率：', tp / (tp + fp))
    # print('1样本召回率：', tp / (tp + fn))
    # print('0样本准确率：', tn / (tn + fn))
    # print('0样本召回率：', tn / (tn + fp))
    return pipe, tp / (tp + fp), tp / (tp + fn), tn / (tn + fn), tn / (tn + fp)


if __name__ == '__main__':
    trainSamples = get_samples(train0_file='data/train_0.txt', train1_file='data/train_1.txt')
    trainLabel, trainSet, testLabel, testSet = split_train_test(train_samples=trainSamples, test_ratio=0.2)
    trained_pipe, r1, r2, r3, r4 = svm_train('linear', trainLabel, trainSet, testLabel, testSet)
    save_model(pipe_model=trained_pipe)
    print(r1, r2, r3, r4)
