import numpy as np
import cv2
# 读取视频文件
cap = cv2.VideoCapture('1.mp4')
# 或者电影每秒的帧数
fps = cap.get(cv2.CAP_PROP_FPS)
# 判断视频是否一直打开
i = 0
txtx = open('3MP4/test.txt','w')
while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break
    cv2.imwrite('3MP4/'+'0'*(4-len('%d'%i))+'%d'%i+'.png',frame)
    i += 1
    txtx.write('3MP4/'+'0'*(4-len('%d'%i))+'%d'%i+'.png ')
    txtx.write('3MP4/' + '0' * (4 - len('%d' % i)) + '%d' % i + '.png')
    txtx.write('\n')
# 清除缓存退出
cv2.destroyAllWindows()