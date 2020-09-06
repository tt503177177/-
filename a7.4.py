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

import os
import numpy as np
import tensorflow as tf
from decimal import Decimal, ROUND_HALF_UP
#要考慮udp的設定大小
#防止tensorflow.python.framework.errors_impl.InternalError: Failed to create session.
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #使用编号为0的gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 
config.gpu_options.allow_growth = True 

gpu_options = tf.GPUOptions(allow_growth=True)   #這台要加這兩句
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))#這台要加這兩句
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge #
sys.path.append('/home/streak/object_detection_ros')
from utils import label_map_util
from utils import visualization_utils as vis_util

local_address = ('', 9000)
# Create a UDP connection that we'll send the command to
#socket.socket(參數1,參數2)   通訊格式            廣播
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind to the local address and port
sock.bind(local_address)

# IP and port of Tello
tello_address = ('192.168.10.1', 8889)

MODEL_NAME = '/home/streak/object_detection_ros/20200320_sea_detection'
PATH_TO_CKPT = '/home/streak/object_detection_ros/20200320_sea_detection/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/streak/object_detection_ros/training/label_sea.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)



image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#這個
pub = rospy.Publisher('image', Image, queue_size=1)
rospy.init_node('talker', anonymous=True)
bridge = CvBridge()

'''#
msg = 'command'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)
time.sleep(0.5)
response0, ip_address = sock.recvfrom(128)
response0 = response.decode(encoding='utf-8')
'''


msg = 'command'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)
time.sleep(0.5)
msg = 'streamon'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)
time.sleep(0.2)

video = cv2.VideoCapture('udp://@0.0.0.0:11111')  #'udp://@0.0.0.0:11111'
ret = video.set(3,480) #weight
ret = video.set(4,360)   #height   1280*720 

c=1 
cc = 5   
p = 0                      #用於計算第n張
timeFF = 30
timeF  = 10                    #每10幀
t = 0
time.sleep(1)

#'''
#tello use
for i in range(350):
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)    
    
    if(cc%timeFF == 0):  
    
        (boxes, scores, classes, num) = sess.run(            
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(            
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        
        print('initialize')        
        cv2.imshow('initialize', frame)
        
    cc = cc + 1

cv2.destroyAllWindows()
#'''

'''#
msg = 'command'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)
time.sleep(4)

msg = 'takeoff'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)
time.sleep(4)

msg = 'up 40'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)
time.sleep(0.5)

msg = 'battery?'.encode(encoding="utf-8")
sent = sock.sendto(msg, tello_address)  
time.sleep(0.3)
response11, ip_address = sock.recvfrom(128)
'''

ret = video.set(3,960) #weight
ret = video.set(4,720) 

while(True):
    
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    if(c%101 == 0):
        msg = 'command'.encode(encoding="utf-8")
        sent = sock.sendto(msg, tello_address)

    if(c%timeF == 0):
  
        (boxes, scores, classes, num) = sess.run(            
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(            
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.75)
        #frame = cv2.resize(frame, (480, 360))
        pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))

        #s_boxes = boxes[scores > confident]
        #s_scores = scores[scores > confident]
        #s_classes = classes[scores > confident]

    c = c + 1


    if cv2.waitKey(1) == ord('q'):   # Press 'q' to quit
        msg = 'stop'.encode(encoding="utf-8")
        sent = sock.sendto(msg, tello_address)    
        time.sleep(0.2)
        msg = 'land'.encode(encoding="utf-8")
        sent = sock.sendto(msg, tello_address) 
        sock.close()        
        break

