import os
import os.path as ops
import tensorflow as tf
import numpy as np
import utils
import sys
import SANet_model
import glob
import cv2
import time
import matplotlib
matplotlib.use('Agg') #　not show up just write into disk
from matplotlib import pyplot as plt
import copy
import train_data_provider
import validate_data_provider

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# dataset_dir = "./data_for_test/"
dataset_dir = "Input/"
output_density_map = "./cmap_output"
batch_size = 1
epoch = 1
loss_c_weight = 0.001
loss_weight = 12

if __name__ == "__main__":
    validate_dataset = validate_data_provider.DataSet(ops.join(dataset_dir, 'test.txt'))

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
    estimated_density_map = SANet_model.scale_aggregation_network(x)
    estimated_counting = tf.reduce_sum(estimated_density_map, reduction_indices=[1, 2, 3], name="crowd_counting")
    #　set tf saver
    saver = tf.train.Saver()


    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 　for restore
        # weights_path = '/home/lyt/Desktop/counting_experiment/reference_model/SANet_not_official/SANet_tensorflow/checkpoint_dir/counting_epoch169.ckpt'
        weights_path = './checkpoint_dir/counting_epoch393.ckpt'
        saver.restore(sess=sess, save_path=weights_path)

        for i in range(epoch):
            i = i + 393
            #　visualize density map
            if i == 393:
                density_map_dir = output_density_map + "/epoch" + str(i)
                density_map_dir_seperate = output_density_map + "/epoch" + str(i) + "_seperate"
                if not ops.exists(density_map_dir):
                    os.makedirs(density_map_dir)
                if not ops.exists(density_map_dir_seperate):
                    os.makedirs(density_map_dir_seperate)

                figure, (origin, pred) = plt.subplots(1, 2, figsize=(14, 4))
                # image_validate_num = 13
                image_validate_num = len(glob.glob(dataset_dir+'*.png'))
                for m in range(image_validate_num):               
                    density_map_path = density_map_dir +"/d_map" + str(m + 1) + ".png"          
                    gt_imgs_9patches, height, weight, gt_imgs = validate_dataset.next_batch(batch_size)
                    a = time.time()
                    estimated_density_map_9patches = sess.run(estimated_density_map, feed_dict={x: gt_imgs_9patches})
                    density_map_full = np.zeros((batch_size, height, weight, 1))
                    density_map_full[0][0:(height//4+height//8), 0:(weight//4+weight//8), :] = estimated_density_map_9patches[0][0:(height//4+height//8), 0:(weight//4+weight//8), :]
                    density_map_full[0][0:(height//4+height//8), (weight//4+weight//8):(weight//2+weight//8), :] = estimated_density_map_9patches[1][0:(height//4+height//8), weight//8:(weight//2-weight//4+weight//8), :]
                    density_map_full[0][0:(height//4+height//8), (weight//2+weight//8):weight, :] = estimated_density_map_9patches[2][0:(height//4+height//8), weight//8:(weight-weight//2), :]

                    density_map_full[0][(height//4+height//8):(height//2+height//8), 0:(weight//4+weight//8), :] = estimated_density_map_9patches[3][height//8:(height//2-height//4+height//8), 0:(weight//4+weight//8), :]
                    density_map_full[0][(height//4+height//8):(height//2+height//8), (weight//4+weight//8):(weight//2+weight//8), :] = estimated_density_map_9patches[4][height//8:(height//2-height//4+height//8), weight//8:(weight//2-weight//4+weight//8), :]
                    density_map_full[0][(height//4+height//8):(height//2+height//8), (weight//2+weight//8):weight, :] = estimated_density_map_9patches[5][height//8:(height//2-height//4+height//8), weight//8:(weight-weight//2), :]

                    density_map_full[0][(height//2+height//8):height, 0:(weight//4+weight//8), :] = estimated_density_map_9patches[6][height//8:(height-height//2), 0:(weight//4+weight//8), :]
                    density_map_full[0][(height//2+height//8):height, (weight//4+weight//8):(weight//2+weight//8), :] = estimated_density_map_9patches[7][height//8:(height-height//2), weight//8:(weight//2-weight//4+weight//8), :]
                    density_map_full[0][(height//2+height//8):height, (weight//2+weight//8):weight, :] = estimated_density_map_9patches[8][height//8:(height-height//2), weight//8:(weight-weight//2), :] 

                    est_counting = sess.run(estimated_counting, feed_dict={estimated_density_map: density_map_full})
                    b = time.time()
                    
                    origin.imshow(gt_imgs[0][:, :, ::-1]) #BGR to RGB
                    origin.set_title('origin Image')
                    pred.imshow(np.squeeze(density_map_full), cmap=plt.cm.jet)
                    pred.set_title('estimated_density_map {}'.format(est_counting))
                    plt.savefig(density_map_path)

                    density_map_path_seperate = density_map_dir_seperate + "/" + str(m + 1) + "_dmap" +  ".png" 
                    plt.imsave(density_map_path_seperate, np.squeeze(density_map_full), cmap=plt.cm.jet)
                    gt_ = copy.deepcopy(gt_imgs[0])

                    cv2.putText(gt_, 'frame: %d fps: %.2f num: %d' % (m, b - a, est_counting[0]),
                               (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
                    cv2.imshow('vido',gt_)
                    cv2.waitKey(1)