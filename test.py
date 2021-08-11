# 读入数据
import pandas as pd
DATA = pd.read_csv('rushdata.csv', index_col=0)

data = DATA

# 打乱数据集，通过打乱索引的方式
import random
rlist = [i for i in range(len(data))]
random.shuffle(rlist)

# 构造输入
import numpy as np
alldata = np.zeros((len(data),5))
for i in range(len(data)):
    alldata[i] = np.array([data['rn'][i],data['rvip'][i],data['rvag'][i],data['rq1'][i],data['rq2'][i]])

# 分割数据集
traindata = alldata[0:120000]
testdata = alldata[120000:150000]

# 定义并作图
import matplotlib.pyplot as plt
def plot3D(testdata, y_pred):
    x = []
    y = []
    z = []
    for i in range(len(testdata)):
        x.append(testdata[i][0])
        y.append(testdata[i][1])
        z.append(testdata[i][3])

    plt.figure("3D Scatter", facecolor="lightgray")
    ax3d = plt.subplot(projection="3d")  # 创建三维坐标

    plt.title('3D Scatter', fontsize=20)
    ax3d.set_xlabel('pin', fontsize=10)
    ax3d.set_ylabel('t', fontsize=10)
    ax3d.set_zlabel('reqline', fontsize=10)
    plt.tick_params(labelsize=10)

    ax3d.scatter(x, y, z, s=1, c=y_pred, cmap="jet", marker="o")

    plt.show()

# 1.对原始点作图
orin_l = [1 for i in range(len(testdata))]
plot3D(testdata, orin_l)

# 2.Kmeans
from sklearn.cluster import KMeans
from sklearn import metrics
kmeans = KMeans(n_clusters=2, n_init=1, init=np.array([[0,0,0,0,0],[800,800,800,800,1]]),tol=0.0001)
kmeans.fit(traindata)
y_pred = kmeans.predict(testdata)
plot3D(testdata, y_pred)

# 3.孤立森林
# 孤立森林
from sklearn.ensemble import IsolationForest
# 创建模型，n_estimators：int，可选（默认值= 100），集合中的基本估计量的数量
model_isof = IsolationForest(n_estimators=100,max_features=2,contamination='auto')
# 计算有无异常的标签分布
model_isof.fit(traindata)
outlier_label = model_isof.predict(testdata)
for i in range(len(outlier_label)):
    if outlier_label[i] == 1:
        outlier_label[i] = 0
    else :
        outlier_label[i] = 1
# 作图
plot3D(testdata, outlier_label)

# 4.HBOS
from pyod.models.hbos import HBOS
model_hbos = HBOS(n_bins=10,alpha=0.1,tol=0.5,contamination=0.2)
model_hbos.fit(traindata)
hbos_pred = model_hbos.predict(testdata)
plot3D(testdata, hbos_pred)