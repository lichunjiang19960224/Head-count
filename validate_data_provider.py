#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-3 下午4:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : data_provider.py
# @IDE: PyCharm
"""
训练数据生成器
"""
import os.path as ops
import h5py
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as CM


class DataSet(object):
    """
    实现数据集类
    """
    def __init__(self, dataset_info_file):
        """
        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_list = self._init_dataset(dataset_info_file)
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """
        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                gt_img_list.append(info_tmp[1])
                gt_label_list.append(info_tmp[0])
        return gt_img_list, gt_label_list

    # @staticmethod
    def _generate_training_pathches(self, gt_img, patch_nums):
        """
        在标签图像和原始图像上随机扣取图像对
        :param gt_img:
        :param label_img:
        :param patch_nums:
        :param patch_size:
        :return:
        """
        height = gt_img.shape[0]
        weight = gt_img.shape[1]
        
        gt_img_patches = []
        gt_img_patch_1 = gt_img[0:height//2, 0:weight//2, :]
        gt_img_patch_2 = gt_img[0:height//2, weight//4:(weight//2+weight//4), :]
        gt_img_patch_3 = gt_img[0:height//2, weight//2:weight, :]
        gt_img_patch_4 = gt_img[height//4:(height//2+height//4), 0:weight//2, :]
        gt_img_patch_5 = gt_img[height//4:(height//2+height//4), weight//4:(weight//2+weight//4), :]
        gt_img_patch_6 = gt_img[height//4:(height//2+height//4), weight//2:weight, :]
        gt_img_patch_7 = gt_img[height//2:height, 0:weight//2, :]
        gt_img_patch_8 = gt_img[height//2:height, weight//4:(weight//2+weight//4), :]
        gt_img_patch_9 = gt_img[height//2:height, weight//2:weight, :]
        gt_img_patches.append(gt_img_patch_1)
        gt_img_patches.append(gt_img_patch_2)
        gt_img_patches.append(gt_img_patch_3)
        gt_img_patches.append(gt_img_patch_4)
        gt_img_patches.append(gt_img_patch_5)
        gt_img_patches.append(gt_img_patch_6)
        gt_img_patches.append(gt_img_patch_7)
        gt_img_patches.append(gt_img_patch_8)
        gt_img_patches.append(gt_img_patch_9)

        return gt_img_patches


    def next_batch(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]

            img_patches = []
            gt_images = []

            for index, gt_img_path in enumerate(gt_img_list):
                gt_image = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
       
                height = gt_image.shape[0]
                weight = gt_image.shape[1]

                gt_images.append(gt_image)

                gt_image_patches = self._generate_training_pathches(gt_img=gt_image, patch_nums=9)
                for index, gt_image_patch in enumerate(gt_image_patches):
                    img_patches.append(gt_image_patch)

            self._next_batch_loop_count += 1
            return img_patches, height, weight, gt_images


if __name__ == '__main__':
    val = DataSet('Data/Gan_Derain_Dataset/train/train.txt')
    a1, a2, a3 = val.next_batch(6)
    import matplotlib.pyplot as plt
    for index, gt_image in enumerate(a1):
        plt.figure('test_{:d}_src'.format(index))
        plt.imshow(gt_image[:, :, (2, 1, 0)])
        plt.figure('test_{:d}_label'.format(index))
        plt.imshow(a2[index][:, :, (2, 1, 0)])
        plt.figure('test_{:d}_mask'.format(index))
        plt.imshow(a3[index], cmap='gray')
        plt.show()

