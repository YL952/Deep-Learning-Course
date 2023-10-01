import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import PIL.Image as image

paths ="./stones.jpg"
X = plt.imread(paths)
X = np.array(X)
#print(X.shape)
#print(X[0][0])
shape = row, col, dim = X.shape

X_ = X.reshape(-1,3)#(将矩阵化为2维，才能使用聚类)

def kmeans(X, n):
    kmeans = KMeans(n_clusters = n)
    kmeans.fit(X)
    y = kmeans.predict(X)
    return y

plt.figure(figsize=(12, 6))
plt.subplot(2,3,1)
plt.title("original picture")
plt.imshow(X)
plt.xticks([])
plt.yticks([])
for t in range(2, 7):
    index = '23' + str(t)
    plt.subplot(int(index))
    label = kmeans(X_,t)
    print("label.shape=",label.shape)
    # get the label of each pixel
    label = label.reshape(row, col)
    # create a new image to save the result of K-Means
    pic_new = image.new("RGB", (col, row))#定义的是图像大小为y*x*3的图像，这里列在前面行在后面
    for i in range(col):
        for j in range(row):
                if label[j][i] == 0:
                    pic_new.putpixel((i, j), (123, 104, 238))#填写的是位置为（j,i）位置的像素，列和行也是反的
                elif label[j][i] == 1:
                    pic_new.putpixel((i, j), (220, 20, 60))
                elif label[j][i] == 2:
                    pic_new.putpixel((i, j), (0, 255, 0))
                elif label[j][i] == 3:
                    pic_new.putpixel((i, j), (60, 0, 220))
                elif label[j][i] == 4:
                    pic_new.putpixel((i, j), (249, 219, 87))
                elif label[j][i] == 5:
                    pic_new.putpixel((i, j), (131, 175, 155))
                elif label[j][i] == 6:
                    pic_new.putpixel((i, j), (220, 87, 18))
    title = "k="+str(t)
    plt.title(title)
    plt.imshow(pic_new)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.xticks([])
    plt.yticks([])
plt.show()