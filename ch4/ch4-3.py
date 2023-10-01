from time import time
import logging
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import cv2 as cv
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, predictiveLabel)

    # Normalized mutual information
    NMI = metrics.normalized_mutual_info_score(trueLabel, predictiveLabel)

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)

    return ACC, NMI, ARI


path = './face_images'
IMAGE_SIZE = (180, 200)
image_SIZE = 360
IMAGE_COLUMN = 20
IMAGE_ROW = 10
to_image = Image.new('RGB', (IMAGE_COLUMN * image_SIZE, IMAGE_ROW * image_SIZE))
# 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db


def createDatabase(path):
    # 查看路径下所有文件
    TrainFiles = os.listdir(path)
    # 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db
    Train_Number = len(TrainFiles)
    T = []
    Y = []
    # 把所有图片转为1-D并存入T中
    for k in range(0, Train_Number):
        Trainneed = os.listdir(path+ '/' +TrainFiles[k])
        Trainneednumber = len(Trainneed)
        for i in range(0, Trainneednumber):
            img = Image.open(path + '/' + TrainFiles[k]+ '/' + Trainneed[i]).resize(
                (image_SIZE, image_SIZE), Image.ANTIALIAS)
            to_image.paste(img, ((i) * image_SIZE, (k) * image_SIZE))
            image = cv.imread(path + '/' + TrainFiles[k]+ '/' + Trainneed[i],cv.IMREAD_GRAYSCALE)
            image = cv.cvtColor(image,cv.COLOR_BAYER_GB2GRAY)
            image = cv.resize(image, IMAGE_SIZE)
            # print(img)
        # 转为1-D
            image = image.reshape(image.size, 1)
            T.append(image)
            Y.append(k)
    T = np.array(T)
    Y = np.array(Y)
    return T ,Y

n_samples, h, w = 200, 200, 180
X,Y = createDatabase(path)
n_features = X.shape[1]
X_ = X.reshape(n_samples, h*w)
n_components = 10
t0 = time()

k_model = k_means(X_, n_clusters=10)
cluster_center_circle = k_model[0]
cluster_label_circle = k_model[1]
ACC,NMI,ARI = clusteringMetrics(Y, cluster_label_circle)
print("Evaluating...")
print("------------")
print("ACC:  ", ACC)
print("NMI:  ", normalized_mutual_info_score(Y,cluster_label_circle))
print("ARI:  ", adjusted_rand_score(Y,cluster_label_circle))
print("------------")
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_)
# print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

# print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_)
# print("done in %0.3fs" % (time() - t0))


def plot_gallery(images, titles, h, w, n_row=1, n_col=10):
    """Helper function to plot a gallery of portraits"""
    # plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.figure(figsize=(1.8 * n_col, 3 * n_row))
    # plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(10):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow( images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



eigenface_titles = ["%d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.figure()
plt.imshow(to_image)

ACCS= []
NMIS = []
ARIS = []


for i in range(1, 9):
    n_components = i
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_)
    # print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    # print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_)
    # print("done in %0.3fs" % (time() - t0))
    k_model = k_means(X_train_pca, n_clusters=10)
    cluster_center_circle = k_model[0]
    cluster_label_circle = k_model[1]
    ACC, NMI, ARI = clusteringMetrics(Y, cluster_label_circle)
    ACCS.append(ACC)
    NMIS.append(NMI)
    ARIS.append(ARI)
fig, ax = plt.subplots()
bar_width = 0.2

opacity = 0.4
error_config = {'ecolor': '0.3'}
index = np.arange(8)
rects1 = ax.bar(index, ACCS, bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='ACC')

rects2 = ax.bar(index + 0.04 + bar_width, NMIS, bar_width,
                alpha=opacity, color='m', error_kw=error_config,
                label='NMI')

rects3 = ax.bar(index + 0.08 + 2*bar_width, ARIS, bar_width,
                alpha=opacity, color='g', error_kw=error_config,
                label='ARI')

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8'))
ax.legend()

plt.show()