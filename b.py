# -*- coding: utf-8 -*-
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import socket
import time
from PIL import Image
from PIL import ImageEnhance
from PyQt5 import QtCore, QtGui, QtWidgets
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image #
import gc

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge
'''
sys.path.append('/home/streak/object_detection_ros')
from utils import label_map_util

'''



xx = str('%')



class video(QMainWindow):
    def __init__(self):
    #def __init__(self, cam): 有用cam的時候
        super(video,self).__init__()

        # 初始化传入的摄像头句柄为实例变量,并得到摄像头宽度和高度
        self.resize(1600, 1024)
        self.setWindowTitle('gui')    

        # 设置GUI窗口的位置和尺寸
        #self.setGeometry(300, 200, self.w+200, self.h+200)

        # 打印得到的摄像头图像的宽和高
        #print(self.w, self.h)
        self.vF = QLabel()
        self.setCentralWidget(self.vF)
        self.vF.setGeometry(QtCore.QRect(160, 152, 960, 720)) 
        #位置x y  長 寬  位置是看左上角的點

        
        self.label_2 = QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(1410, 890, 191, 131))
        #self.label_2.setText("")
        #圖片寫法好像不同？
        pix = QPixmap('aa.png')
        self.label_2.setPixmap(pix)
        self.label_2.setObjectName("label_2")
        self.label_2.setScaledContents(True)
       

        # 设置视频显示在窗口中间,否则可以注释掉
        self.vF.setAlignment(Qt.AlignCenter)
        self.pushButton_camera = QPushButton(self)
        self.pushButton_camera.setGeometry(QtCore.QRect(30, 40, 200, 70))
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.pushButton_camera.setText("開啟相機")

        
        self.battery_textBrowser1 = QTextBrowser(self)
        self.battery_textBrowser1.setGeometry(QtCore.QRect(830, 20, 220, 40))
        self.battery_textBrowser1.setObjectName("battery_textBrowser1")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.battery_textBrowser1.setFont(font)
        self.battery_textBrowser1.setText('大無人機電量： 0%s' %xx)  #battery
        self.battery_textBrowser1.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.battery_textBrowser1.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.battery_textBrowser = QTextBrowser(self)
        self.battery_textBrowser.setGeometry(QtCore.QRect(480, 20, 220, 40))
        self.battery_textBrowser.setObjectName("battery_textBrowser")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.battery_textBrowser.setFont(font)
        self.battery_textBrowser.setText('小無人機電量： 0%s' %xx)  #battery
        self.battery_textBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.battery_textBrowser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)


        self.battery_textBrowser2 = QTextBrowser(self)
        self.battery_textBrowser2.setGeometry(QtCore.QRect(480, 90, 220, 40))
        self.battery_textBrowser2.setObjectName("battery_textBrowser2")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.battery_textBrowser2.setFont(font)
        self.battery_textBrowser2.setText('小無人機座標：(0,0)' )  #battery
        self.battery_textBrowser2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.battery_textBrowser2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        

        self.battery_textBrowser3 = QTextBrowser(self)
        self.battery_textBrowser3.setGeometry(QtCore.QRect(830, 90, 220, 40))
        self.battery_textBrowser3.setObjectName("battery_textBrowser3r")
        font = QtGui.QFont()
        font.setPointSize(16)
        self.battery_textBrowser3.setFont(font)
        self.battery_textBrowser3.setText('大無人機座標：(0,0)' )  #battery
        self.battery_textBrowser3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.battery_textBrowser3.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)


        # 设置定时器 每250毫秒执行实例的play函数以刷新图像

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.load)
        self._timer.start(250)

    def load(self):
        rospy.Subscriber("image", Image, self.callback)


    def callback(self, imgmsg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        img = cv2.resize(img, (960, 720))
        self.vF.setPixmap(QPixmap.fromImage(
            QImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                960,
                720,
                13)))


  

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    app = QApplication(sys.argv)

    win = video()
    win.show()      
    sys.exit(app.exec_())