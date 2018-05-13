import os
from PIL import Image
import numpy as np
import time
import cv2

# import pyautogui
# import re
"""
将截图保存的图片进行融合拼接成一个长图

"""
# TODO optimize 当检查到底部的图片和下一张图片（局部相似，或者是距离在误差之内即可）相似之后，
# TODO 从下一界面，将对应的图像从头开始取出，覆盖上一张的底部部分图片

# 图片压缩后的大小
width_i = 200
height_i = 300

# 每行每列显示图片数量
line_max = 10
row_max = 10

# 参数初始化
all_path = []
num = 0
pic_max = line_max * row_max

dirName = os.getcwd()

images_dir = "./images"

all_path = [images_dir + "/" + file for file in os.listdir(images_dir) if file.endswith("png")]

# for root, dirs, files in os.walk(dirName):
#     for file in files:
#         if "jpeg" in file:
#             all_path.append(os.path.join(root, file))
print(all_path)


def get_image_height(path):
    image = Image.open(path)
    return image.size


WIDTH, HEIGHT = get_image_height(all_path[0])
print(HEIGHT)

TOP_PADDING = 106
BOTTOM_Y = 597
BOTTOM_PADDING = HEIGHT - BOTTOM_Y
SUB_IMAGE_HEIGHT = BOTTOM_Y - TOP_PADDING

Y = 0  # BOTTOM_Y

toImage = Image.new('RGB', (WIDTH, HEIGHT))
num = len(all_path)


# stitcher = cv2.createStitcher(False)
# foo = cv2.imread("./1524553607031.png")
# bar = cv2.imread("./1524553608306.png")
# result = stitcher.stitch((foo, bar))
# cv2.imwrite("./result.jpg", result[1])

# 556 596
# exit(0)


def concat_by_channel(image1, image2):
    for i in range(0, image2.shape[1]):
        print(i)


def distance2D(bbx1, bbx2):
    aa = bbx1 - bbx2
    # print("distance is ", aa)
    mean = np.mean(bbx1)
    sqr = np.square(bbx1 - bbx2)
    sum = np.sum(sqr)

    bb = np.sqrt(sum)/mean
    # print(" bb is ", bb)
    return bb


def distance3D(bbx1, bbx2):
    aa = bbx1.reshape(-1) - bbx2.reshape(-1)
    # print("distance is ", aa)
    mean = np.mean(bbx1)
    sqr = np.square(aa)
    sum = np.sum(sqr)
    bb = np.sqrt(sum)/mean
    # print(" bb is ", bb)
    return bb


def concate_image(img1, img2, from_y, to_y, overwrite=False,compare_H=36):
    """
    将image进行拼接
    :param img1: source image
    :param img2: to merger image
    :param from_y: from index of height axis
    :param to_y: to index of height axis
    :param overwrite: overwrite mode or concat mode
    :param compare_H:

    :return: concate image

    """
    img = []
    if from_y == 0 and overwrite:
        print("overwrite ...")
        img1[img1.shape[0] - (to_y-from_y):, :, :] = img2[0:, :, :]
        img = img1
    elif from_y == 0:
        print("concate from zero")
        # img1[from_y:to_y, :] = img2
        img = np.vstack((img1, img2))
        print("merge from 0{} to {}".format(0, to_y))
    elif from_y < to_y:  # concat then replace
        img = np.vstack((img1, img2[from_y:, :,:]))  # TODO concat_H
        img[img.shape[0] - (to_y - from_y)-compare_H:, :, :] = img2[from_y-compare_H:,:, :]
        print("merge from {} to {}".format(from_y, to_y))
    else:
        print("error invalid from_y({}) > to_y({})".format(from_y, to_y))
        raise AssertionError(from_y >= to_y)

    # final = cv2.hconcat((img1, img2))
    # cv2.imshow('final', final)
    # 纵向拼接
    # im = cv2.vconcat(src=(img1, img2), dst=img1)
    # cv2.imshow("mergedimage", img1)

    # image = np.concatenate((gray1, gray2))
    # 纵向连接=np.vstack((gray1, gray2))
    # 横向连接image = np.concatenate([gray1, gray2], axis=1)
    return img
def concate_or_replace_image(image1, image2):
    """
    遍历y 方向，并在x方向采样，取出几个关键区域进行比对，得到的距离和
    如果说是距离为0，直接跳出循环，得到对应y

    方案1
    否则继续进行得到所有的y方向的距离索引 WARN 此处可能因为相似对很高导致的图片距离小于一定的距离机会误判断
    并排序，得到距离最小的y的索引从该位置进行拼接（ 或者说是尝试直接跳过，认为是不重叠的直接进行拼接，因为图像不变性）

    :param image1:
    :param image2:
    :return:
    """

    print("concate_or_replace_image ...")
    b, g, r = cv2.split(image1)

    b2, g2, r2 = cv2.split(image2)
    # 并取出底部的左边部分的局部图片 t1, t2, t3, t4
    N = 12
    # STEP = (95 - 10) // (N + 1)
    STEP = (95 - 10) // (N + 1)
    PAD = 3  # 采样宽度
    compare_H = 39  # 底部的高度
    y1 = image1.shape[0] - compare_H
    y2 = image2.shape[0] - compare_H
    image2H = image2.shape[0]
    distances = []
    for y in range(y2 + 1):
        dist = 0
        dist2 = 0
        for i in range(0, N):
            im_sub1 = image1[y1:y1 + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD, :]
            im_sub2 = image2[y:y + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD, :]
            bx1 = b[y1:y1 + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]
            bx2 = b2[y:y + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]

            gx1 = g[y1:y1 + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]
            gx2 = g2[y:y + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]

            rx1 = r[y1:y1 + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]
            rx2 = r2[y:y + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]
            # if y == y2 - 1:
            #     print("=====")
            dist2 += distance3D(im_sub1,im_sub2)
            dist += distance2D(bx1, bx2)+distance2D(gx1, gx2)+distance2D(rx1, rx2)
        # distances.append(dist)
        distances.append(dist2)
        if dist == 0:
            print("find nearest 0 distance index:", y)
            break

    min_index = np.argmin(distances)
    min_dis = distances[min_index]
    DISTANCES = 60.272  # 23075.png 59.272158265002744 6696.png dis is:95.90278204692277
    print("min_dis is : ",min_dis)
    # print("distances is ===== \n", distances)
    #  y2 need to be fixed by add compare_H == image2H ！！！
    if min_dis == 0:
        print("distance is zero in index ", min_index)
        if min_index < y2:
            print("1. concate image from {} to {}", min_index, image2H)
            # TODO  fix from min to image2H (conca then replace with par2)
            image = concate_image(image1, image2, min_index + compare_H, image2H,compare_H=compare_H)
            # cv2.imshow('image', image)
            cv2.imwrite(result_images.format(time.time()), image)
        else:
            print("2. overwrite image from {} to {}".format(0, image2H))
            image = concate_image(image1, image2, 0, image2H, overwrite=True)
            cv2.imwrite(result_images.format(time.time()), image)
            # cv2.imshow('image', image)
    elif len(distances) == y2 + 1 and min_dis < DISTANCES:  # 遍历到最后，发现距离很小，认为是重合的，
        image = concate_image(image1, image2, min_index + compare_H, image2H, overwrite=False)
        cv2.imwrite(result_images.format(time.time()), image)
        print("3. iterate to the end y:", y2, " and find min distance ")

    elif len(distances) == y2 + 1 and min_dis >= DISTANCES:  # 遍历到最后发现，没有任何重合的，则进行拼接操作
        image = concate_image(image1, image2, 0, image2H, overwrite=False)
        cv2.imwrite(result_images.format(time.time()), image)
        print("4. concate the imges from {} to {}", 0, image2H)
    else:
        print(" other case !!! ")
    print("y_index is ===== ", min_index, min_dis)
    # time.strptime(a, "%Y-%m-%d %H:%M:%S")
    return image
    # 将 ti 进行 与b2的纵向遍历，横向位置比对，计算出距离 ，并求和


# def concate_or_replace_image(image1, image2):
#     """
#     遍历y 方向，并在x方向采样，取出几个关键区域进行比对，得到的距离和
#     如果说是距离为0，直接跳出循环，得到对应y
#
#     方案1
#     否则继续进行得到所有的y方向的距离索引 WARN 此处可能因为相似对很高导致的图片距离小于一定的距离机会误判断
#     并排序，得到距离最小的y的索引从该位置进行拼接（ 或者说是尝试直接跳过，认为是不重叠的直接进行拼接，因为图像不变性）
#
#     :param image1:
#     :param image2:
#     :return:
#     """
#
#     print("concate_or_replace_image ...")
#     b, g, r = cv2.split(image1)
#
#     b2, g2, r2 = cv2.split(image2)
#     # 并取出底部的左边部分的局部图片 t1, t2, t3, t4
#     N = 15
#     STEP = (95 - 10) // (N + 1)
#     PAD = 3  # 采样宽度
#     compare_H = 39  # 底部的高度
#     y1 = image1.shape[0] - compare_H
#     y2 = image2.shape[0] - compare_H
#     image2H = image2.shape[0]
#     distances = []
#     for y in range(y2 + 1):
#         dist = 0
#         for i in range(0, N):
#             bx1 = b[y1:y1 + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]
#             bx2 = b2[y:y + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD]
#             # if y == y2 - 1:
#             #     print("=====")
#             dist += distance(bx1, bx2)
#         distances.append(dist)
#         if dist == 0:
#             print("find nearest 0 distance index:", y)
#             break
#     print()
#     min_index = np.argmin(distances)
#     min_dis = distances[min_index]
#     DISTANCES = 37.8608
#     print("min_dis is : ",min_dis)
#     # print("distances is ===== \n", distances)
#     #  y2 need to be fixed by add compare_H == image2H ！！！
#     if min_dis == 0:
#         print("distance is zero in index ", min_index)
#         if min_index < y2:
#             print("1. concate image from {} to {}", min_index, image2H)
#             # TODO  fix from min to image2H (conca then replace with par2)
#             image = concate_image(image1, image2, min_index + compare_H, image2H,compare_H=compare_H)
#             # cv2.imshow('image', image)
#             cv2.imwrite(result_images.format(time.time()), image)
#         else:
#             print("2. overwrite image from {} to {}".format(0, image2H))
#             image = concate_image(image1, image2, 0, image2H, overwrite=True)
#             cv2.imwrite(result_images.format(time.time()), image)
#             # cv2.imshow('image', image)
#     elif len(distances) == y2 + 1 and min_dis < DISTANCES:  # 遍历到最后，发现距离很小，认为是重合的，
#         image = concate_image(image1, image2, min_index + compare_H, image2H, overwrite=False)
#         cv2.imwrite(result_images.format(time.time()), image)
#         print("3. iterate to the end y:", y2, " and find min distance ")
#
#     elif len(distances) == y2 + 1 and min_dis >= DISTANCES:  # 遍历到最后发现，没有任何重合的，则进行拼接操作
#         image = concate_image(image1, image2, 0, image2H, overwrite=False)
#         cv2.imwrite(result_images.format(time.time()), image)
#         print("4. concate the imges from {} to {}", 0, image2H)
#     else:
#         print(" other case !!! ")
#     print("y_index is ===== ", min_index, min_dis)
#     # time.strptime(a, "%Y-%m-%d %H:%M:%S")
#     return image
#     # 将 ti 进行 与b2的纵向遍历，横向位置比对，计算出距离 ，并求和


if __name__ == '__main__':
    result_images = "./result/meged{}.jpg"

    image_conc = []
    image_b = []
    for i in range(0, num):
        print("process image {} ".format(all_path[i]))
        img = cv2.imread(all_path[i])
        cut_img = img[TOP_PADDING:TOP_PADDING + SUB_IMAGE_HEIGHT, 0:WIDTH]
        # b, g, r = cv2.split(cut_img)
        if i == 0:
            print("===== init first image ======")
            image_conc = cut_img
            # image_b = b
        else:
            # 对下面的一张图片进行局部比对
            if '608306' in all_path[i]:
                print(" process  1524553610222")

            image_conc = concate_or_replace_image(image_conc, cut_img)
            print()
            # break


# image_res = []
# image_corners = []

#
#     image = cv2.imread(all_path[i])
#     # 截取 中间有效的图片
#     image = image[TOP_PADDING:BOTTOM_Y, 0:WIDTH]
#
#     # 获取该部分的图片角点
#
#     # 转化为灰度float32类型进行处理
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     img_gray = np.float32(img_gray)
#
#     # 得到角点坐标向量
#     goodfeatures_corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=100, qualityLevel=0.1, minDistance=15,
#                                                    corners=100)
#     goodfeatures_corners = np.int0(goodfeatures_corners)
#     # In [45]: array.sort(key=lambda x:x[1])#lambda x:x[1]返回list的第二个数据
#     goodfeatures_corners.sort(axis=-1)
#     goodfeatures_corners2 = np.sort(goodfeatures_corners, axis=2, kind='quicksort', order=None)
#     print(goodfeatures_corners)
#
#
#     if i == 0:
#         image_res = image
#         cv2.imwrite('lena3.jpg', image)
#         image_corners = goodfeatures_corners
#
#     else:
#         # image2 = Image.open(all_path[i])
#         # 比较该角点和记录的角点位置是不是一致
#         for x in image_corners:
#             print(x)
#
#
#             # toImage.paste(image1, (0, Y))
#             # Y += SUB_IMAGE_HEIGHT
#
# print(toImage.size)
#
# toImage.save('merged.png')
#
# exit(-1)

#
# for i in range(0, len(all_path)):
#
#     for j in range(0, line_max):
#         pic_fole_head = Image.open(all_path[num])
#         width, height = pic_fole_head.size
#         tmppic = pic_fole_head.resize((width_i, height_i))
#         loc = (int(i % line_max * width_i), int(j % line_max * height_i))
#         # print("第" + str(num) + "存放位置" + str(loc))
#         toImage.paste(tmppic, loc)
#         num += 1
#         if num >= len(all_path):
#             print("breadk")
#             break
#     if num >= pic_max:
#         break
