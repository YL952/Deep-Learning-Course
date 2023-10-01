import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
###
# 聚类\分类精度模板、三大指标模板
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


n_clusters = 10
image_resize_row = 180
image_resize_cols = 200
image_date_row = 1
image_date_cols = 10
h, w = 200, 180

# 后缀
suffix_list = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

# 图像类别(标签)
dataset_list = ["admars", "ahodki", "ajflem", "ajones", "ajsega",
                "anpage", "asamma", "asewil", "astefa", "drbost"]


def calculEigenfaces():
    # 数据集路径
    src_dir = ".//face_images//"
    # 获取图片以及对应类别（彩色）
    # imgs, labels = getGrayImgAndLabel(src_dir)
    imgs, labels = getColorImgAndLabel(src_dir)
    # imgs, labels = getColorImgAndLabel_onedir(src_dir)

    images = np.array(imgs)  # 存储原始矩阵
    # print(images.shape)

    # 将图片转换为数组(彩色)
    # arr = convertGrayImageToArrays(imgs)
    arr = convertColorImageToArrays(imgs)

    # 先训练PCA模型,n_components为降到的维数
    pca = PCA(n_components=10)
    pca.fit(arr)

    # 返回测试集和训练集降维后的数据集
    arr_pca = pca.transform(arr)
    # print("arr_pca的形状：", arr_pca.shape)
    plt.figure(figsize=(1.8 * 10, 3 * 1))
    for index in range(10):
        #feature_img = convert_array_to_image(arr_pca[:,index:index+1], 200, 180)
        feature_img = convert_array_to_color_image(arr_pca[:, index:index + 1], 200, 180)
        #feature_img = arr_pca[:, index].reshape(200, 180)
        # print("mean_arr's shape : {}".format(feature_img.shape))
        # print("type(mean_img) : {}".format(type(feature_img)))
        plt.subplot(1, 10, index + 1)
        plt.imshow(feature_img, cmap=plt.cm.gray)
        plt.title(str(index + 1))
        plt.xticks(())
        plt.yticks(())
        file_name = str(index) + '.png'
        cv2.imwrite(file_name, feature_img)

    plt.show()

def getGrayImgAndLabel(src_dir):
    """
    加载训练集
    :param src_dir: 数据集路径
    :return:
    """
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    # 获取子文件夹名
    catelist = os.listdir(src_dir)
    # 遍历子文件夹
    k = 0
    for catename in catelist:
        # 设置子文件夹路径
        cate_dir = os.path.join(src_dir, catename)
        # 获取子文件名
        filelist = os.listdir(cate_dir)
        # 遍历所有文件名
        for filename in filelist:
            # 设置文件路径
            file_dir = os.path.join(cate_dir, filename)
            # 判断文件名是否为图片格式
            if not os.path.splitext(filename)[1] in suffix_list:
                print(file_dir, "is not an image")

                continue
            # endif
            # # 读取灰度图
            imgs.append(cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE))
            # 读取相应类别
            labels.append(catename)
    # endfor
    # endfor
    return imgs, labels


def getColorImgAndLabel_onedir(src_dir):
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    # 获取文件名
    filelist = os.listdir(src_dir)
    # 遍历所有文件名
    for filename in filelist:
        # 设置文件路径
        file_dir = os.path.join(src_dir, filename)
        # 判断文件名是否为图片格式
        if not os.path.splitext(filename)[1] in suffix_list:
            print(file_dir, "is not an image")

            continue
        # endif
        # # 读取彩色图像：三通道，unit8
        imgs.append(cv2.imread(file_dir))
        # 读取相应类别
        labels.append(filelist)
    return imgs, labels


def getColorImgAndLabel(src_dir):
    """
    加载训练集
    :param src_dir: 数据集路径
    :return:
    """
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    # 获取子文件夹名
    k = 0
    catelist = os.listdir(src_dir)
    # 遍历子文件夹
    for catename in catelist:
        # 设置子文件夹路径
        cate_dir = os.path.join(src_dir, catename)
        # 获取子文件名
        filelist = os.listdir(cate_dir)
        # 遍历所有文件名
        for filename in filelist:
            # 设置文件路径
            file_dir = os.path.join(cate_dir, filename)
            # 判断文件名是否为图片格式
            if not os.path.splitext(filename)[1] in suffix_list:
                print(file_dir, "is not an image")

                continue
            # endif
            # # 读取彩色图像：三通道，unit8
            imgs.append(cv2.imread(file_dir))
            # 读取相应类别
            # labels.append(catename)
            labels.append(k)

        k = k + 1

    labels = np.array(labels)
    # endfor
    # endfor
    return imgs, labels


# end of getImgAndLabel

# 将图像（灰度）数据变为一列
def convertGrayImageToArray(img):
    img_arr = []
    height, width = img.shape[:2]
    # 遍历图像
    for i in range(height):
        img_arr.extend(img[i, :])
    # endfor
    return img_arr


# 将每个（灰度）图像变为一列
def convertGrayImageToArrays(imgs):
    # 初始化数组
    arr = []
    # 遍历每个图像
    for img in imgs:
        arr.append(convertGrayImageToArray(img))
    # endfor
    return np.array(arr).T


# 将图像（彩色）数据变为一列
def convertColorImageToArray(img):
    img_arr = []
    height, width, channel = img.shape[:3]

    # 遍历图像
    for h in range(height):
        for w in range(width):
            img_arr.extend(img[h][w][:])

    # print(np.array(img_arr).shape)

    return img_arr


# 将每个（彩色）图像变为一列
def convertColorImageToArrays(imgs):
    # 初始化数组
    arr = []
    # 遍历每个图像
    for img in imgs:
        arr.append(convertColorImageToArray(img))
    # endfor
    return np.array(arr).T


# 计算均值数组
def compute_mean_array(arr):
    # 获取维数(行数),图像数(列数)
    dimens, nums = arr.shape[:2]
    # 新建列表
    mean_arr = []
    # 遍历维数
    for i in range(dimens):
        # 求和每个图像在该字段的值并平均
        aver = int(sum(arr[i, :]) / float(nums))
        mean_arr.append(aver)
    # endfor
    return np.array(mean_arr)


# end of compute_mean_array


# 将数组转换为对应图像
def convert_array_to_image(arr, height=256, width=256):
    img = []
    for i in range(height):
        img.append(arr[i * width:i * width + width])
    # endfor
    return np.array(img)


# 将数组转换为对应图像
def convert_array_to_color_image(arr, height=256, width=256):
    img = arr.reshape(200, 180, 3)
    # endfor
    return np.array(img)


# 计算图像和平均图像之间的差值
def compute_diff(arr, mean_arr):
    return arr - mean_arr


# end of compute_diff

# 计算每张图像和平均图像之间的差值
def compute_diffs(arr, mean_arr):
    diffs = []
    dimens, nums = arr.shape[:2]
    for i in range(nums):
        diffs.append(compute_diff(arr[:, i], mean_arr))
    # endfor
    return np.array(diffs).T


# end of compute_diffs

# 计算协方差矩阵的特征值和特征向量，按从大到小顺序排列
# arr是预处理图像的矩阵，每一列对应一个减去均值图像之后的图像
def compute_eigenValues_eigenVectors(arr):
    arr = np.array(arr)
    # 计算arr'T * arr
    temp = np.dot(arr.T, arr)
    eigenValues, eigenVectors = np.linalg.eig(temp)
    # 将数值从大到小排序
    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    # 特征向量按列排
    eigenVectors = eigenVectors[:, idx]
    return eigenValues, np.dot(arr, eigenVectors)


# end of compute_eigenValues_eigenVectors

# 计算图像在基变换后的坐标(权重)
def compute_weight(img, vec):
    return np.dot(img, vec)


# end of compute_weight

# 计算图像权重
def compute_weights(imgs, vec):
    dimens, nums = imgs.shape[:2]
    weights = []
    for i in range(nums):
        weights.append(compute_weight(imgs[:, i], vec))
    return np.array(weights)


# end of compute_weights

# 计算两个权重之间的欧式距离
def compute_euclidean_distance(wei1, wei2):
    # 判断两个向量的长度是否相等
    if not len(wei1) == len(wei2):
        print('长度不相等')

        os._exit(1)
    # endif
    sqDiffVector = wei1 - wei2
    sqDiffVector = sqDiffVector ** 2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances ** 0.5
    return distance


# end of compute_euclidean_distance

# 计算待测图像与图像库中各图像权重的欧式距离
def compute_euclidean_distances(wei, wei_test):
    weightValues = []
    nums = wei.shape
    print(nums)

    for i in range(nums[0]):
        weightValues.append(compute_euclidean_distance(wei[i], wei_test))
    # endfor
    return np.array(weightValues)


# end of compute_euclidean_distances


def image_compose_clusterdata(clusterdata):
    """
    使用PIL.Image将聚类后的图像拼接成一张大图
    :param clusterdata: 数据格式：List
    :return:
    """
    # list转为array
    print("用于聚类的数据类型：", type(clusterdata), "数据维度：", clusterdata.shape)  # 打印数据维度与类型

    # 创建一个新图 (mode,(width,height))
    to_image = Image.new('RGB', (image_date_cols * image_resize_cols, image_date_row * image_resize_row))
    print("to_image的大小", to_image.size)

    # 遍历数组，将所有数据格式从array转为Image
    for row in range(len(clusterdata) - 1):
        for cols in range(len(clusterdata[row])):
            from_image = clusterdata[row][cols].reshape(image_resize_cols, image_resize_row, 3)
            from_image = Image.fromarray(cv.cvtColor(from_image, cv.COLOR_BGR2RGB))
            to_image.paste(from_image, (cols * image_resize_cols, row * image_resize_row))

    plt.imshow(to_image)
    plt.title("ACC=%f, ARI=%f, NMI=%f" % (ACC, ARI, NMI))
    plt.show()

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

calculEigenfaces()