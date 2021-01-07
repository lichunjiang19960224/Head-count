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

import random

class DataSet(object):
    """
    实现数据集类
    """
    def __init__(self, dataset_info_file):
        """
        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_list = self._init_dataset(dataset_info_file)
        self._random_dataset()
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
        # return 8*gt_img_list, 8*gt_label_list
        return 1*gt_img_list, 1*gt_label_list

    def _random_dataset(self):
        """
        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_list.append(self._gt_label_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_list = new_gt_label_list

    # @staticmethod
    def _generate_training_pathches(self, gt_img, label_img, patch_nums):
        """
        在标签图像和原始图像上随机扣取图像对
        :param gt_img:
        :param label_img:
        :param patch_nums:
        :param patch_size:
        :return:
        """
        # if gt_img.shape != label_img.shape:
        #     label_img = cv2.resize(label_img, dsize=(gt_img.shape[1], gt_img.shape[0]),
        #                            interpolation=cv2.INTER_NEAREST)
        height = gt_img.shape[0]
        weight = gt_img.shape[1]
        
        gt_img_patches = []
        label_img_patches = []
       
        # np.random.seed(1234)
        for i in range(patch_nums):
            # seed_x = np.random.randint(int(patch_size / 2 + 1), img_w - int(patch_size / 2) - 1)
            # seed_y = np.random.randint(int(patch_size / 2 + 1), img_h - int(patch_size / 2) - 1)
            left_y = random.randint(0, height // 2)
            left_x = random.randint(0, weight // 2)
            gt_img_patch = gt_img[left_y:left_y + height // 2, left_x:left_x + weight // 2, :]
            label_img_patch = label_img[left_y:left_y + height // 2, left_x:left_x + weight // 2, :]

            # for the downsample and upsample in the model
            h_patch = gt_img_patch.shape[0]
            w_patch = gt_img_patch.shape[1]
            h_residual = h_patch % 16   # 4 downsamples and 4 upsamples
            w_residual = w_patch % 16  
            if h_residual!=0:
                gt_img_patch = gt_img_patch[0:h_patch-h_residual, :, :]
                label_img_patch = label_img_patch[0:h_patch-h_residual, :, :]
            if w_residual!=0:
                gt_img_patch = gt_img_patch[:, 0:w_patch-w_residual, :]
                label_img_patch = label_img_patch[:, 0:w_patch-w_residual, :]

            # flip
            if random.randint(0, 1)%2==0:
                gt_img_patch = gt_img_patch[:, ::-1, :]
                label_img_patch = label_img_patch[:, ::-1, :]

            # # BGR2GRAY
            # if random.randint(0, 1)%2==0:
            #     gt_img_patch = cv2.cvtColor(gt_img_patch, cv2.COLOR_BGR2GRAY)
            #     gt_img_patch = np.expand_dims(gt_img_patch, axis=-1)
            #     temp_array = np.zeros((gt_img_patch.shape[0], gt_img_patch.shape[1], 3), dtype=int)
            #     for i in range(3):
            #         temp_array[:,:,i:i+1] = gt_img_patch
            #     gt_img_patch = temp_array
      
            gt_img_patches.append(gt_img_patch)
            label_img_patches.append(label_img_patch)
        return gt_img_patches, label_img_patches

    def nomalize(self, input_tensor, mean, std):
        output = None
        for i, means in enumerate(mean):
            input_tensor[:,:,i] = (input_tensor[:,:,i]-means)/std[i]
        output = input_tensor
        return output

    def next_batch(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        assert len(self._gt_label_list) == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels = []

            for index, gt_img_path in enumerate(gt_img_list):
                gt_image = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
                # cv2.imshow('lala.jpg',gt_image)
                # cv2.waitKey(0)
                
                label_file = h5py.File(gt_label_list[index])
                label_image = np.asarray(label_file['density']) 
                label_image = np.expand_dims(label_image, axis=-1)
    
                gt_image_patches, label_image_patches = self._generate_training_pathches(gt_img=gt_image,label_img=label_image,patch_nums=1)
                for index, gt_image_patch in enumerate(gt_image_patches):
                    gt_imgs.append(gt_image_patch)
                    gt_labels.append(label_image_patches[index])

            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels


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

