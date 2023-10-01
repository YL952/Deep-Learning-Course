from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
import os
import pandas as pd
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred):

    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind1, ind2 = linear_assignment(w.max() - w)
    return sum([w[i, j] for i in ind1 for j in ind2]) * 1.0 / y_pred.size
def getinfo():
    # 获取文件并构成向量
    #预测值为1维，把一张图片的三维压成1维，那么n张图片就是二维
    global total_photo
    file = os.listdir('./face_images/')
    i = 0
    for subfile in file:
        photo = os.listdir(r'.\face_images\\' + subfile)  #文件路径自己改
        for name in photo:
            photo_name.append(r'.\face_images\\'+ subfile+'\\'+name)
            target.append(i)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path)
        photo = photo.reshape(1, -1)
        photo = pd.DataFrame(photo)
        total_photo = total_photo.append(photo, ignore_index=True)
    total_photo = total_photo.values
def kmeans():
    clf = KMeans(n_clusters=10)
    clf.fit(total_photo)
    y_predict = clf.predict(total_photo)
    centers = clf.cluster_centers_
    result = centers[y_predict]
    result = result.astype("int64")
    result = result.reshape(200, 200, 180, 3)#图像的矩阵大小为200,180,3
    return result,y_predict

def draw():
    fig,ax  = plt.subplots(nrows=10,ncols=20,sharex = True,sharey = True,figsize = [15,8],dpi = 80)
    plt.subplots_adjust(wspace = 0,hspace = 0)
    count = 0
    for i in range(10):
        for j in range(20):
            ax[i,j].imshow(result[count])
            count += 1

    plt.xticks([])
    plt.yticks([])
    plt.show()

def score():
    ACC = cluster_acc(target, y_predict)  # y 真实值 y_predict 预测值
    NMI = normalized_mutual_info_score(target, y_predict)
    ARI = adjusted_rand_score(target, y_predict)
    print(" ACC = ", ACC)
    print(" NMI = ", NMI)
    print(" ARI = ", ARI)
    
if __name__ == '__main__':
    photo_name = []
    target = []
    total_photo = pd.DataFrame()
    getinfo()
    result,y_predict = kmeans()
    score()
    draw()