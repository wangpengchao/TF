import numpy
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.metrics import recall_score    # 召回率
from sklearn.metrics import precision_score  # 精准率

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

estimators = [
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=0.1, loss="hinge", random_state=42))
]

pipe = Pipeline(estimators)
pipe.fit(trainData, trainLabel)
# print(pipe.predict([testData[0]]))
# print(testLabel[0])

# 总准确率和召回率
# accuracyScore = accuracy_score(testLabel, pipe.predict(testData))
# print('accuracy score: ', accuracyScore)
# recallScore = recall_score(testLabel, pipe.predict(testData))
# print('recall score: ', recallScore)
# precisionScore = precision_score(testLabel, pipe.predict(testData))
# print('precision score:', precisionScore)

# 分开计算0和1样本的 准确率 召回率
# print(testLabel)
# print('==')
# print(pipe.predict(testData))
predictResult = pipe.predict(testData)

tp = 0
fp = 0
fn = 0
tn = 0

for i in range(len(testLabel)):
    if (testLabel[i] == 1) and (predictResult[i] == 1):
        tp = tp+1
    if (testLabel[i] == 0) and (predictResult[i] == 1):
        fp = fp+1
    if (testLabel[i] == 1) and (predictResult[i] == 0):
        fn = fn+1
    if (testLabel[i] == 0) and (predictResult[i] == 0):
        tn = tn+1

print('1样本准确率：', tp/(tp+fp))
print('1样本召回率：', tp/(tp+fn))
print('0样本准确率：', tn/(tn+fn))
print('0样本召回率：', tn/(tn+fp))
