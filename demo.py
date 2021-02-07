from mainwindow import Ui_MainWindow
import sys, validate_data_provider,cv2,copy,os
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtGui import *
import tensorflow as tf
import SANet_model,glob,time
import os.path as ops
from matplotlib import pyplot as plt
from PyQt5.QtCore import QTimer

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
dataset_dir = "Input/"
output_density_map = "./cmap_output"
batch_size = 1
epoch = 1
loss_c_weight = 0.001
loss_weight = 12

class query_window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.__init()
        self.i = 0
        # self.timer = QTimer()
        #         # # 定时器结束，触发showTime方法
        #         # self.timer.start(10)
        #         # self.timer.timeout.connect(self.det)
        #         # self.ui.
        #         # self.ui.Start.clicked.connect(self.query_formula)
        #         # 给button 的 点击动作绑定一个事件处理函数
    def __init(self):
        self.validate_dataset = validate_data_provider.DataSet(ops.join(dataset_dir, 'test.txt'))
        figure, (self.origin, self.pred) = plt.subplots(1, 2, figsize=(14, 4))
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
        self.estimated_density_map = SANet_model.scale_aggregation_network(self.x)
        self.estimated_counting = tf.reduce_sum(self.estimated_density_map, reduction_indices=[1, 2, 3], name="crowd_counting")
        # 　set tf saver
        saver = tf.train.Saver()

        weights_path = './checkpoint_dir/counting_epoch393.ckpt'
        saver.restore(sess=self.sess, save_path=weights_path)
        self.image_validate_num = len(glob.glob(dataset_dir + '*.png'))

        self.density_map_dir = output_density_map + "/epoch" + str(0)
        self.density_map_dir_seperate = output_density_map + "/epoch" + str(0) + "_seperate"
        if not ops.exists(self.density_map_dir):
            os.makedirs(self.density_map_dir)
        if not ops.exists(self.density_map_dir_seperate):
            os.makedirs(self.density_map_dir_seperate)
        self.ui.text.append('finish init!')

    def det(self):
        if self.i < self.image_validate_num:
            density_map_path = self.density_map_dir + "/d_map" + str(self.i + 1) + ".png"
            gt_imgs_9patches, height, weight, gt_imgs = self.validate_dataset.next_batch(batch_size)
            a = time.time()
            estimated_density_map_9patches = self.sess.run(self.estimated_density_map, feed_dict={self.x: gt_imgs_9patches})
            density_map_full = np.zeros((batch_size, height, weight, 1))
            density_map_full[0][0:(height // 4 + height // 8), 0:(weight // 4 + weight // 8), :] = \
            estimated_density_map_9patches[0][0:(height // 4 + height // 8), 0:(weight // 4 + weight // 8), :]
            density_map_full[0][0:(height // 4 + height // 8), (weight // 4 + weight // 8):(weight // 2 + weight // 8),
            :] = estimated_density_map_9patches[1][0:(height // 4 + height // 8),
                 weight // 8:(weight // 2 - weight // 4 + weight // 8), :]
            density_map_full[0][0:(height // 4 + height // 8), (weight // 2 + weight // 8):weight, :] = \
            estimated_density_map_9patches[2][0:(height // 4 + height // 8), weight // 8:(weight - weight // 2), :]

            density_map_full[0][(height // 4 + height // 8):(height // 2 + height // 8), 0:(weight // 4 + weight // 8),
            :] = estimated_density_map_9patches[3][height // 8:(height // 2 - height // 4 + height // 8),
                 0:(weight // 4 + weight // 8), :]
            density_map_full[0][(height // 4 + height // 8):(height // 2 + height // 8),
            (weight // 4 + weight // 8):(weight // 2 + weight // 8), :] = estimated_density_map_9patches[4][
                                                                          height // 8:(
                                                                                      height // 2 - height // 4 + height // 8),
                                                                          weight // 8:(
                                                                                      weight // 2 - weight // 4 + weight // 8),
                                                                          :]
            density_map_full[0][(height // 4 + height // 8):(height // 2 + height // 8),
            (weight // 2 + weight // 8):weight, :] = estimated_density_map_9patches[5][
                                                     height // 8:(height // 2 - height // 4 + height // 8),
                                                     weight // 8:(weight - weight // 2), :]

            density_map_full[0][(height // 2 + height // 8):height, 0:(weight // 4 + weight // 8), :] = \
            estimated_density_map_9patches[6][height // 8:(height - height // 2), 0:(weight // 4 + weight // 8), :]
            density_map_full[0][(height // 2 + height // 8):height,
            (weight // 4 + weight // 8):(weight // 2 + weight // 8), :] = estimated_density_map_9patches[7][
                                                                          height // 8:(height - height // 2),
                                                                          weight // 8:(
                                                                                      weight // 2 - weight // 4 + weight // 8),
                                                                          :]
            density_map_full[0][(height // 2 + height // 8):height, (weight // 2 + weight // 8):weight, :] = \
            estimated_density_map_9patches[8][height // 8:(height - height // 2), weight // 8:(weight - weight // 2), :]

            est_counting = self.sess.run(self.estimated_counting, feed_dict={self.estimated_density_map: density_map_full})
            b = time.time()

            self.origin.imshow(gt_imgs[0][:, :, ::-1])  # BGR to RGB
            self.origin.set_title('origin Image')
            self.pred.imshow(np.squeeze(density_map_full), cmap=plt.cm.jet)
            self.pred.set_title('estimated_density_map {}'.format(est_counting))
            plt.savefig(density_map_path)

            density_map_path_seperate = self.density_map_dir_seperate + "/" + str(self.i + 1) + "_dmap" + ".png"
            plt.imsave(density_map_path_seperate, np.squeeze(density_map_full), cmap=plt.cm.jet)
            gt_ = copy.deepcopy(gt_imgs[0])
            cv2.putText(gt_, 'frame: %d fps: %.2f num: %d' % (self.i, b - a, est_counting[0]),
                        (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            jiqir = cv2.cvtColor(gt_, cv2.COLOR_BGR2RGB)
            self.ui.image.setScaledContents(True)
            Q_img = QImage(jiqir,
                           jiqir.shape[0],
                           jiqir.shape[1],
                           jiqir.shape[1] * 3, QImage.Format_RGB888)
            self.ui.image.setPixmap(QtGui.QPixmap(Q_img))
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = query_window()
    window.show()
    sys.exit(app.exec_())