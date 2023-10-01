from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score


# Sklearn中的make_circles方法生成训练样本
data, color = datasets.make_circles(n_samples=800, noise=0.1, factor=0.2)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], s=80, marker="o", c=color, cmap=plt.cm.RdYlBu, edgecolors='black')
# c=Y_circle划分两种标签数据的颜色
plt.title('data by make_circles')

estimator = KMeans(n_clusters=2)  # 构造聚类器
y = estimator.fit_predict(data)  # 聚类
centroids = estimator.cluster_centers_  # 获取聚类中心

acc = accuracy_score(color, y)
nmi = normalized_mutual_info_score(color, y)
ari = adjusted_rand_score(color, y)

print(f"ACC: {acc:.3f}")
print(f"NMI: {nmi:.4f}")
print(f"ARI: {ari:.4f}")

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], s=80, marker="o", edgecolors='black', c=y, cmap=plt.cm.RdYlBu)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, marker="*", edgecolors='black', color='green')
plt.title('K-means (n_clusters=2)')

plt.show()

