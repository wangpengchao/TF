import numpy
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.metrics import recall_score    # 召回率

train0 = numpy.loadtxt(fname='data/train_0.txt', delimiter=',')
train1 = numpy.loadtxt(fname='data/train_1.txt', delimiter=',')

# 随机抽取300条数据
numpy.random.shuffle(train0)
newTrain0 = train0[0:300]

train = numpy.vstack((newTrain0, train1))  # 按照纵轴堆叠成为新的矩阵

numpy.random.shuffle(train)  # 将数据打乱

trainLabel = train[:400, 0].astype(numpy.int)
trainSet = train[0:400, 1:]  # 400条数据作为 训练集

testLabel = train[400:, 0].astype(numpy.int)
testSet = train[400:, 1:]  # 165条数据作为 测试集

# 数据集进行归一化操作(-1~1)
trainData = normalize(trainSet, axis=0)
testData = normalize(testSet, axis=0)


# 使用网格搜索确定最优参数
estimatorsG = [
    ("poly_features", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(loss="hinge", random_state=42))
    # ("svm_clf", SVC())  # 未使用网格搜索之前确定的参数 kernel='poly', degree=2, gamma=5, C=1
]
pipe = Pipeline(estimatorsG)
gridParameter = dict(
                      poly_features__degree=[1, 2, 3, 4, 5, 6],
                      # svm_clf__kernel=['linear', 'poly', 'rbf'],
                      svm_clf__C=[0.01, 0.1, 1, 10, 100],
                      )
gridSearch = GridSearchCV(pipe, param_grid=gridParameter, cv=5)
gridSearch.fit(trainData, trainLabel)
print('best parameter', gridSearch.best_params_)

accuracyScore = accuracy_score(testLabel, gridSearch.predict(testSet))
print('accuracy score: ', accuracyScore)
recallScore = recall_score(testLabel, gridSearch.predict(testSet))
print('recall score: ', recallScore)


# estimators = [
#         ("scaler", StandardScaler()),
#         ("svm_clf", SVC(kernel='poly', C=10, gamma=5, coef0=1, degree=2 ))
# ]
# pipe = Pipeline(estimators)
#
# pipe.fit(trainData, trainLabel)
# accuracyScore = accuracy_score(testLabel, pipe.predict(testSet))
# print('accuracy score: ', accuracyScore)
# recallScore = recall_score(testLabel, pipe.predict(testSet))
# print('recall score: ', recallScore)