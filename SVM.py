import numpy
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.metrics import recall_score    # 召回率

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
trainData = normalize(trainSet, axis=0)
testData = normalize(testSet, axis=0)

estimators = [("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))]

pipe = Pipeline(estimators)

pipe.fit(trainData, trainLabel)
# print(pipe.predict([testData[0]]))
# print(testLabel[0])

accuracyScore = accuracy_score(testLabel, pipe.predict(testData))
print('accuracy score: ', accuracyScore)
recallScore = recall_score(testLabel, pipe.predict(testData))
print('recall score: ', recallScore)