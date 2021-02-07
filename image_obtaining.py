import numpy as np
import cv2
import os, shutil
# 读取视频文件
cap = cv2.VideoCapture('1.mp4')
# 或者电影每秒的帧数
dir = 'Input//'
fps = cap.get(cv2.CAP_PROP_FPS)
# 判断视频是否一直打开
i = 0
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)
txtx = open(dir + 'test.txt','w')
flag = 0
while (cap.isOpened()):
    success, frame = cap.read()
    if flag == 2:
        flag = flag + 1
        continue
    if success == False:
        break
    flag = 0
    cv2.imwrite(dir+'0'*(4-len('%d'%i))+'%d'%i+'.png',frame)
    i += 1
    txtx.write(dir+'0'*(4-len('%d'%i))+'%d'%i+'.png ')
    txtx.write(dir + '0' * (4 - len('%d' % i)) + '%d' % i + '.png')
    txtx.write('\n')
