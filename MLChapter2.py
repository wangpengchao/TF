import numpy
import os
import tarfile
import pandas
from six.moves import urllib  # 兼容python2和Python3的库

# 随机种子，使代码的运行结果稳定，后面会有更加合适的代码，使每次选取的结果稳定
# numpy.random.seed(42)

import matplotlib
import matplotlib.pyplot as plot

matplotlib.rc('axes', labelsize = 14)
matplotlib.rc('xtick', labelsize = 12)
matplotlib.rc('ytick', labelsize = 12)

# print(help(matplotlib.rc))

# 设置保存图片的路径
PROJECT_ROOT_DIR = '.'
CHARTER_ID = 'end_to_end_project'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images', CHARTER_ID)


def save_fig(figure_id, tight_layout=True, figure_extension='png', resultion=300):
    '''
    存储图片
    :param figure_id:
    :param tight_layout:
    :param figure_extension:
    :param resultion:
    :return:
    '''
    path = os.path.join(IMAGES_PATH, figure_id + '.' + figure_extension)
    print('保存图片',figure_id)
    if tight_layout:
        plot.tight_layout()
    plot.savefig(path, format=figure_extension, dpi=resultion)


# 注意系统路径分隔符和网址分隔符是相反的，所以应当将两分开设定
# 网络URL
download_root = 'https://github.com/wangpengchao/handson-ml/tree/master/'
housing_url = download_root + 'datasets/housing/housing.tgz'

# 本地存储
housing_path = os.path.join('datasets', 'housing')

# print(housing_path)
# print(housing_url)


def fetch_housing_data( housing_url=housing_url, housing_path=housing_path ):
    tgz_path = os.path.join(housing_path, "housing.tgz")

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        # 这块下载文件的代码有问题，位置原因导致下载的文件异常而不能解压，我尝试使用迅雷直接下载后进行打开，也是错误的；而下载的整个GitHub的项目中的文件是可以正确打开的
        urllib.request.urlretrieve(housing_url, tgz_path)

    # 读取文件之后如果不能解压（即打开失败），考虑一下检查下文件是否正确，是否是文件下载出了问题
    housing_tgz = tarfile.open(name=tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# 执行一次将压缩包解压即可
# fetch_housing_data()


def loadHouseData(housPath = housing_path):
    """
    读取使用 fetch_housing_data() 函数解压之后的csv文件
    :param housPath: housing文件的路径
    :return: housing.csv函数
    """
    csvPath = os.path.join(housPath, 'housing.csv')
    return pandas.read_csv(csvPath)


house = loadHouseData()
# print(house.head())

# 获取数据集的简单描述：实体数量、列数、每列的名称、每一列的数据类型等信息
# print(house.info())

# 如果有某些列的数值是重复的，这意味着它有可能是一个分类属性，使用如下函数可以查看有多少种分类存在，每种类别下面有多少个区域
# print(house['ocean_proximity'].value_counts())

# 针对每一列，列出这一列的count mean std min max 25% 50% 75%等描述信息
# print(house.describe())

# 如下调用的是 pandas.DataFrame.hist 函数生成直方图，bins表示每个图片中有多少个小直方
# house.hist(bins=50, figsize=(20, 15))
# plot.show()

# 画出单独属性的直方图
# house['latitude'].hist(bins=10, figsize=(20,20))


# 如下代码仅供说明怎么从样本中分离 训练集 和 测试集，sklearn中train_test_split() 函数用于分离
def splitTrainTest(data, testRatio):
    '''
    从data数据中拆分出 训练集 和 测试集 并返回
    :param data: 待分离的数据
    :param testRatio: 测试集占的比例
    :return:训练集 测试集
    '''
    shuffledIndices = numpy.random.permutation(len(data))  # shuffle函数是原地打乱，不是对副本进行打乱

    print('===',shuffledIndices)
    print(len(shuffledIndices))

    testSetSize = int(len(data)*testRatio)

    testIndices = shuffledIndices[:testSetSize]
    trainIndices = shuffledIndices[testSetSize:]

    return data.iloc[trainIndices], data.iloc[testIndices]  # iloc使用数字索引提取行，loc使用名称索引提取行


# trainSet, testSet = splitTrainTest(data=house, testRatio=0.2)
# print(len(trainSet), 'train + ', len(testSet), 'test')

from zlib import crc32


def testSetCheck(identifier, testRatio):
    return crc32(numpy.int64(identifier)) & 0xffffffff < testRatio * 2**32


def splitTrainTestById(data, testRatio, id_colum):
    ids = data[id_colum]

    # print(type(ids))  # pandas.core.series.Series 类型
    inTestSet = ids.apply(lambda id_: testSetCheck(id_, testRatio))
    print(type(inTestSet))
    return data.loc[~inTestSet], data.loc[inTestSet]


houseWithId = house.reset_index()  # 重置索引，默认将旧索引添加为列，并使用新的顺序索引

houseWithId['id'] = house['longitude']*1000 + house['latitude']

trainSet, testSet = splitTrainTestById(houseWithId, 0.2, 'index')
print(testSet.head())














