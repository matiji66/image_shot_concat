#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Spring
"""
将手机截图保存的图片进行融合,拼接成一个长图
"""
# TODO optimize 当检查到底部的图片和下一张图片（局部相似，或者是距离在误差之内即可）相似之后，从下一界面，将对应的图像从头开始取出，覆盖上一张的底部部分图片
# TODO 将对应的 top和bottom补全
# TODO 尝试去除 cv2

import os
import numpy as np
import time
import cv2

# 参数初始化
# dirName = os.getcwd()
images_dir = "./images"
all_path = [images_dir + "/" + file for file in os.listdir(images_dir) if file.endswith("png") or file.endswith("jpg")]

print(all_path)
num = len(all_path)


def get_image_height(path):
    image = cv2.imread(path)
    return image.shape


HEIGHT, WIDTH, CHANNEL = get_image_height(all_path[0])
print(HEIGHT)

TOP_PADDING = 106
BOTTOM_Y = 597
BOTTOM_PADDING = HEIGHT - BOTTOM_Y
SUB_IMAGE_HEIGHT = BOTTOM_Y - TOP_PADDING


def concat_by_channel(image1, image2):
    for i in range(0, image2.shape[1]):
        print(i)


def distance2D(bbx1, bbx2):
    aa = bbx1 - bbx2
    # print("distance is ", aa)
    mean = np.mean(bbx1)
    sqr = np.square(bbx1 - bbx2)
    sum = np.sum(sqr)
    bb = np.sqrt(sum) / mean
    # print(" bb is ", bb)
    return bb


def distance3D(bbx1, bbx2):
    aa = bbx1.reshape(-1) - bbx2.reshape(-1)
    mean = np.mean(bbx1)
    sqr = np.square(aa)
    sum = np.sum(sqr)
    bb = np.sqrt(sum) / mean
    # print(" bb is ", bb)
    return bb


def concate_image(img1, img2, from_y, to_y, overwrite=False, compare_H=36):
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
    # final = cv2.hconcat((img1, img2))
    # cv2.imshow('final', final)
    # 纵向拼接
    # im = cv2.vconcat(src=(img1, img2), dst=img1)
    # cv2.imshow("mergedimage", img1)
    # image = np.concatenate((gray1, gray2))
    # 纵向连接=np.vstack((gray1, gray2))
    # 横向连接image = np.concatenate([gray1, gray2], axis=1)

    if from_y == 0 and overwrite:
        print("overwrite ...")
        img1[img1.shape[0] - (to_y - from_y):, :, :] = img2[0:, :, :]
        img = img1
    elif from_y == 0:
        print("concate from zero")
        # img1[from_y:to_y, :] = img2
        img = np.vstack((img1, img2))
        print("merge from 0{} to {}".format(0, to_y))
    elif from_y < to_y:  # concat then replace
        img = np.vstack((img1, img2[from_y:, :, :]))  # TODO concat_H
        img[img.shape[0] - (to_y - from_y) - compare_H:, :, :] = img2[from_y - compare_H:, :, :]
        print("merge from {} to {}".format(from_y, to_y))
    else:
        print("error invalid from_y({}) >= to_y({})".format(from_y, to_y))
        raise AssertionError(from_y >= to_y)
    return img


def concate_or_replace_image(image1, image2):
    """
    遍历y 方向，并在x方向采样，取出几个关键区域进行比对，得到的距离和
    如果说是距离为0，直接跳出循环，得到对应y

    距离不为0 方案:
    采用左右距离比较的方式,如果左边距离在容差之内,认为是重合的部分,如果大于容差,则计算右边部分的距离,进行容差比对,
    如果距离在容差之内,切min_index接近, 认为是重叠部分
    否则不重叠,进行相应的全图拼接

    :param image1:
    :param image2:
    :return:
    """
    print("concate_or_replace_image ...")
    # 并取出底部的左边部分的局部图片 t1, t2, t3, t4
    N = 12
    # STEP = (95 - 10) // (N + 1)
    STEP = (95 - 10) // (N + 1)
    PAD = 5  # 采样宽度

    STEP2 = (270 - 170) // N
    PAD2 = 10
    compare_H = 39  # 底部的高度
    y1 = image1.shape[0] - compare_H
    y2 = image2.shape[0] - compare_H
    image2H = image2.shape[0]
    distances1 = []
    distances2 = []
    for y in range(y2 + 1):
        dist = 0  # 左边部分的距离
        dist2 = 0  # 右边部分的距离
        for i in range(0, N):
            im_sub1 = image1[y1:y1 + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD, :]
            im_sub2 = image2[y:y + compare_H, 10 + (i + 1) * STEP:10 + (i + 1) * STEP + PAD, :]
            dist += distance3D(im_sub1, im_sub2)

            im_sub3 = image1[y1:y1 + compare_H, 170 + i * STEP2:170 + i * STEP2 + 2 * PAD, :]
            im_sub4 = image2[y:y + compare_H, 170 + i * STEP2:170 + i * STEP2 + 2 * PAD, :]
            dist2 += distance3D(im_sub3, im_sub4)
            # dist2 += dist
        distances1.append(dist)
        distances2.append(dist2)
        # 采用 3D distance
        if dist == 0:
            print("find nearest 0 distance index:", y)
            break

    min_index = np.argmin(distances1)
    min_dis1 = distances1[min_index]
    min_index2 = np.argmin(distances2)
    min_dis2 = distances2[min_index2]
    # 距离容差
    DISTANCES1 = 10  # 16.689446591758163  # 60.272  # 23075.png 59.272158265002744 6696.png dis is:95.90278204692277
    DISTANCES2 = 10  # 16.689446591758163  # 60.272  # 23075.png 59.272158265002744 6696.png dis is:95.90278204692277

    print("min_index ", min_index, " min_dis1 is : ", min_dis1)
    print("min_index2 ", min_index2, " min_dis2 is : ", min_dis2)
    # print("distances is ===== \n", distances)
    #  y2 need to be fixed by add compare_H == image2H ！！！
    if min_dis1 == 0:
        print("distance is zero in index ", min_index)
        if min_index < y2:
            print("1. concate image from {} to {}", min_index, image2H)
            image = concate_image(image1, image2, min_index + compare_H, image2H, compare_H=compare_H)
        else:
            print("2. overwrite image from {} to {}".format(0, image2H))
            image = concate_image(image1, image2, 0, image2H, overwrite=True)
    elif len(distances1) == y2 + 1 and min_dis1 < DISTANCES1: # (distances2[min_index] < DISTANCES2) and ( min_dis1 < DISTANCES1 or abs(min_index2-min_index) < 5):  # 遍历到最后，发现距离很小，认为是重合的，
        image = concate_image(image1, image2, min_index + compare_H, image2H, overwrite=False)
        print("3. iterate to the end y:", y2, " and find min distance ")

    elif len(distances1) == y2 + 1 and min_dis1 >= DISTANCES1 :  # 遍历到最后发现，没有任何重合的，则进行拼接操作
        if distances2[min_index] < DISTANCES2 and abs(min_index2-min_index) < 3:
            image = concate_image(image1, image2, min_index+compare_H, image2H, overwrite=False)
        else:
            image = concate_image(image1, image2, 0, image2H, overwrite=False)
        print("4. concate the imges from {} to {}", 0, image2H)
    else:
        print(" other case return image1 !!! ")
        image = image1
    print("y_index is ===== ", min_index, min_dis1)
    return image
    # 将 ti 进行 与b2的纵向遍历，横向位置比对，计算出距离 ，并求和


if __name__ == '__main__':
    # result_images = "./result/merged{}.jpg"
    result_images = "./result3D/merged{}.jpg"
    image_conc = []
    image_b = []
    for i in range(0, num):
        print("process image {} ".format(all_path[i]))
        img = cv2.imread(all_path[i])
        cut_img = img[TOP_PADDING:TOP_PADDING + SUB_IMAGE_HEIGHT, 0:WIDTH]
        if i == 0:
            print("===== init first image ======")
            image_conc = cut_img
            # image_b = b
        else:
            # 对下面的一张图片进行局部比对
            # if '1524553605795' in all_path[i]:
            #     print(" process  1524553605795")
            image_conc = concate_or_replace_image(image_conc, cut_img)
            cv2.imwrite(result_images.format(time.time()), image_conc)  # write tmp image to file for debug
            print()
            # break  # for test
    # save concatenated image
    cv2.imwrite("merged_final1.jpg", image_conc)
