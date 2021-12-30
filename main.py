from __future__ import division, print_function, absolute_import

from xlwt import *
import sys

import argparse
# from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

import sys
import qdarkstyle
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit
from PIL import Image, ImageDraw, ImageFont#显示汉字用
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from qtpy import QtGui
from mdm import MDM
import tensorflow as tf
from PyQt5.QtGui import QFont
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()

config.gpu_options.allow_growth=True #不全部占满显存, 按需分配

session = tf.Session(config=config) # 设置session KTF.set_session(sess)

###========全局变量 定义开始=======###
# 打开的是摄像头还是本地视频
camera_or_local_flag = 0
# 识别进行中还是暂停识别中，1为摄像头，2位本地摄像头
Start_or_pause_flag = 0

dis_last = [0]*64 #list(range(64))
fcw = [0]*1
fps = 0.0
result_image = ...
writeVideo_flag = True
if writeVideo_flag:
    # Define the codec and create VideoWriter object
    w1 = 1280  # 返回视频的宽
    h1 = 720  # 返回视频的高
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi格式
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')#MP4格式
    # out = cv2.VideoWriter('./output/'+args["input"][43:57]+ "_" + args["class"] + '_output.avi', fourcc, 15, (w, h))
    out = cv2.VideoWriter('./output/' + "FCW_0104" + '_output.avi', fourcc, 30, (w1, h1))
'''
#功能函数，只是用来往图片中显示汉字
#示例 img = cv2ImgAddText(cv2.imread('img1.jpg'), "大家好，我是片天边的云彩", 10, 65, (0, 0, 139), 20)
参数说明：
img：OpenCV图片格式的图片
text：要写入的汉字
left：字符坐标x值
top：字符坐标y值
textColor：字体颜色
：textSize：字体大小
'''
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
# 画半透明矩形框
#输入图片集矩形框，输出画好的图片
def put_mask(image,bbox1,beta):
    # 画出mask
    zeros1 = np.zeros((image.shape), dtype=np.uint8)
    zeros_mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    color=(0,255,0), thickness=-1 ) #thickness=-1 表示矩形框内颜色填充
    # alpha 为第一张图片的透明度
    alpha = 1
    # beta 为第二张图片的透明度
    #beta = 0.2
    gamma = 0
    mask_img = cv2.addWeighted(image, alpha, zeros_mask1, beta, gamma)
    return mask_img

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # 1、总界面框大小 MainWindow
        MainWindow.resize(1850, 820)  # 总界面框
        # 左边界面区域：verticalLayoutWidget    QWidget类
        self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 25, 1280, 720))  # 左边图片框。左上角坐标及宽度高度
        self.verticalLayoutWidget.setStyleSheet('background-color:rgb(55,55,55)')  # 设置做左边框的颜色
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)  # QVBoxLayout类 垂直地摆放小部件
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # 设置左侧、顶部、右侧和底部边距，以便在布局周围使用。
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_ShowPicture = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_ShowPicture.setObjectName("label_ShowPicture")
        self.verticalLayout.addWidget(self.label_ShowPicture)

        # 右边按钮及显示结果字符的一块区域：verticalLayoutWidget_2    QWidget类
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(1350, 50, 450, 700))  # 右边按钮及显示结果字符的大小
        # self.verticalLayoutWidget_2.setStyleSheet('background-color:rgb(155,155,155)')  # 设置做左边框的颜色
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)  # QVBoxLayout类 垂直地摆放小部件
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        # 设置控件间的间距
        self.verticalLayout_2.setSpacing(30)

        # 按钮1 选择图片按钮：pushButton_select_pcture
        self.pushButton_open_camera = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_open_camera.setObjectName("pushButton_open_camera")
        self.verticalLayout_2.addWidget(self.pushButton_open_camera)  # 将按钮1增加到

        # 按钮2 选择视频按钮：pushButton_select_pcture
        self.pushButton_open_Local_video = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_open_Local_video.setObjectName("pushButton_open_Local_video")
        self.verticalLayout_2.addWidget(self.pushButton_open_Local_video)  # 将按钮1增加到
        # 按钮3 开始识别按钮：pushButton_shibie
        self.pushButton_Start_or_pause = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_Start_or_pause.setObjectName("pushButton_Start_or_pause")
        self.verticalLayout_2.addWidget(self.pushButton_Start_or_pause)
        # 按钮4 选择模型：pushButton_shibie
        self.pushButton_save = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_save.setObjectName("pushButton_save")
        self.verticalLayout_2.addWidget(self.pushButton_save)

        # 放“图像识别结果为”这一句话
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)

        # #lable_2放显示结果1
        # self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        # font = QtGui.QFont()
        # font.setPointSize(15)
        # self.label_2.setFont(font)
        # self.label_2.setText("")
        # self.label_2.setObjectName("label_2")
        # self.verticalLayout_2.addWidget(self.label_2)
        # #png = QtGui.QPixmap('E:/testpicture/animal_pic.png')
        #
        # #lable_3放显示结果2
        # self.lable_3 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        # font = QtGui.QFont()
        # font.setPointSize(15)
        # self.lable_3.setFont(font)
        # self.lable_3.setObjectName("label_3")
        # self.verticalLayout_2.addWidget(self.lable_3)
        #
        # #lable_4放显示结果3
        # self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        # font = QtGui.QFont()
        # font.setPointSize(15)
        # self.label_4.setFont(font)
        # self.label_4.setObjectName("label_4")
        # self.verticalLayout_2.addWidget(self.label_4)

        # 表格
        self.tableWidget = QtWidgets.QTableWidget(self.verticalLayoutWidget_2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(6)  # 列数
        self.tableWidget.setRowCount(100)  # 行数
        # 设置表格的行、列标题，如下：
        self.tableWidget.setHorizontalHeaderLabels(['ID', 'Class', 'Point0', 'Point1', 'Point2', 'Point3'])
        # self.tableWidget.setVerticalHeaderLabels(['1', '2'])
        self.verticalLayout_2.addWidget(self.tableWidget)  # 将表格加入到verticalLayout_2空间中，依次罗列

        # 指定列宽
        self.tableWidget.horizontalHeader().resizeSection(0, 40)
        self.tableWidget.horizontalHeader().resizeSection(1, 90)
        self.tableWidget.horizontalHeader().resizeSection(2, 70)
        self.tableWidget.horizontalHeader().resizeSection(3, 70)
        self.tableWidget.horizontalHeader().resizeSection(4, 70)
        self.tableWidget.horizontalHeader().resizeSection(5, 70)
        # self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)#禁止用户编辑，即只读
        # 行、列标题的显示与隐藏。
        # self.tableWidget.horizontalHeader().setVisible(False)#列标题不显示
        self.tableWidget.verticalHeader().setVisible(False)  # 行标题不显示

        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        # 视频显示与计时器，以及关闭识别
        # 建立通信连接
        self.pushButton_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.deep_sort)
        self.pushButton_open_Local_video.clicked.connect(self.pushButton_open_Local_video_click)  #
        self.pushButton_Start_or_pause.clicked.connect(self.pushButton_Start_or_pause_click)
        self.pushButton_save.clicked.connect(self.pushButton_save_click)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        name_picture = 0

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "System"))
        #  self.label_ShowPicture.setText(_translate("MainWindow", "Picture"))
        self.pushButton_open_camera.setText(_translate("MainWindow", "Open Camera"))
        self.pushButton_open_Local_video.setText(_translate("MainWindow", "Select Local Vedio"))
        self.pushButton_Start_or_pause.setText(_translate("MainWindow", "Start"))
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.label.setText(_translate("MainWindow", "Result："))

    # self.label_2.setText("你点击了按钮")

    image = None

    # 打开摄像头
    def button_open_camera_click(self):
        global camera_or_local_flag
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)
                camera_or_local_flag = 1  # 打开了摄像头
                self.pushButton_open_camera.setText(u'Close')
        else:
            self.timer_camera.stop()
            self.cap.release()
            camera_or_local_flag = 0  # 关闭了摄像头
            self.label_ShowPicture.clear()
            self.pushButton_open_camera.setText(u'Open')

    # 选择图片
    def pushButton_select_pcture_click(self):
        filename = QFileDialog.getOpenFileName(None, 'Open file', 'C:/Users/Desktop/testpicture/')
        # 设置标签的图片
        src0 = cv2.imread(filename[0])
        resized0 = cv2.resize(src0, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imwrite("./temp/temp0.jpg", resized0)
        self.label_ShowPicture.setPixmap(QPixmap("./temp/temp0.jpg"))
        print("filename[0]=", filename[0])
        self.image = Image.open(filename[0])

    # img1 = yolo.detect_image(image)
    # img1.show()

    # 选择视频
    def pushButton_open_Local_video_click(self):
        global camera_or_local_flag
        # 打开本地视频，则关闭摄像头的视频
        self.timer_camera.stop()
        self.cap.release()
        camera_or_local_flag = 0  # 关闭了摄像头
        self.label_ShowPicture.clear()
        self.pushButton_open_camera.setText(u'Open Camera')

        if self.timer_camera.isActive() == False:
            print("打开本地视频")
            self.fileName, self.fileType = QFileDialog.getOpenFileName(None, 'Choose file', '', '*.mp4')
            self.cap_Local_video = cv2.VideoCapture(self.fileName)

            # self.timer_camera.start(30)
            camera_or_local_flag = 2  # 打开了本地视频
            #self.pushButton_open_Local_video.setText(u'关闭本地视频')

            # 打开录像后，不进行播放，等点击开始识别后开始播放
            ret, self.frame = self.cap_Local_video.read()
            img = cv2.resize(self.frame, (1280, 720), interpolation=cv2.INTER_AREA)
            height, width, bytesPerComponent = img.shape  # 取彩色图片的长、宽、通道
            bytesPerLine = 3 * width
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            self.label_ShowPicture.setPixmap(QPixmap(pixmap))
        else:
            self.timer_camera.stop()
            self.cap_Local_video.release()
            camera_or_local_flag = 0  # 关闭了摄像头
            self.label_ShowPicture.clear()
            self.pushButton_open_Local_video.setText(u'Open Local Vedio')

    # 开启按钮
    def pushButton_Start_or_pause_click(self):
        global camera_or_local_flag
        global Start_or_pause_flag
        print(camera_or_local_flag, Start_or_pause_flag)
        print(254)
        if camera_or_local_flag == 1: # 摄像头视频
            if Start_or_pause_flag == 0:
                Start_or_pause_flag = 1  # 代表开始识别
                print(258)
                self.timer_camera.start(30)
                self.pushButton_Start_or_pause.setText(u'Runing')
            # self.deep_sort()  # 调用函数

            else:
                Start_or_pause_flag = 0  # 3代表开始识别
                self.timer_camera.stop(30)
                self.pushButton_Start_or_pause.setText(u'Start')
        elif camera_or_local_flag == 2: # 本地视频
            if Start_or_pause_flag == 0:
                Start_or_pause_flag = 1  # 代表开始识别
                self.timer_camera.start(30)
                print(269)
                print(camera_or_local_flag, Start_or_pause_flag)
                self.pushButton_Start_or_pause.setText(u'Runing')
            # self.deep_sort()  # 调用函数

            else:
                Start_or_pause_flag = 0  # 3代表开始识别
                self.timer_camera.stop(30)
                self.pushButton_Start_or_pause.setText(u'Start')

        print(" 2 camera_or_local_flag =%d Start_or_pause_flag =%d", camera_or_local_flag, Start_or_pause_flag)

    def pushButton_save_click(self):
        print("保存成功")

    # 事件函数
    def select_shipin(self):
        global camera_or_local_flag
        global Start_or_pause_flag
        global result_image
        print("选择视频中")

    def deep_sort(self):
        global camera_or_local_flag
        global Start_or_pause_flag
        global result_image
        global counter_zhenhao
        global counter_jueduizhenhao
        global last_frame_num
        global milliseconds
        global seconds
        global last_seconds
        global fps
        global writeVideo_flag
        K0 = 1
        i = int(0)
        ret = False
        if camera_or_local_flag == 1 and Start_or_pause_flag == 0:
            ret, self.frame = self.cap.read()
        elif camera_or_local_flag == 2 and Start_or_pause_flag == 0:  # 打开本地文件，且只显示首帧
            ret, self.frame = self.cap_Local_video.read()

        if camera_or_local_flag == 1 and Start_or_pause_flag == 1:
            ret, self.frame = self.cap.read()
        elif camera_or_local_flag == 2 and Start_or_pause_flag == 1:  # 打开本地文件，且只显示首帧
            ret, self.frame = self.cap_Local_video.read()

        # print(ret, camera_or_local_flag, Start_or_pause_flag)
        Frame_jiange = 2  # 每2帧取一幅图片
        if ret != True:
            print("打开图像失败 or 识别结束")
            out.release()
            MainWindow.close()  # 关闭界面，退出代码
            return
        if Start_or_pause_flag == 1:
            counter_jueduizhenhao += 1
            t1 = time.time()
            frame0 = cv2.resize(self.frame, (1280, 720), interpolation=cv2.INTER_AREA)
            [height0, width0, tongdao0] = frame0.shape
            milliseconds = self.cap_Local_video.get(cv2.CAP_PROP_POS_MSEC)
            seconds = milliseconds / 1000
            if counter_jueduizhenhao == 13:
                print("调试用")
            #if counter_jueduizhenhao % Frame_jiange == 0:  # 取余每间隔 Frame_jiange 帧处理一次



            frame = frame0.copy()

            [height, width, tongdao] = frame.shape

            # image = Image.fromarray(frame)
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs, class_names = yolo.detect_image(image)
            features = encoder(frame, boxs)
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            # NMS：非极大值抑制
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            # 预测：当目标经过移动，通过上一帧的目标框和速度等参数，预测出当前帧的目标框位置和速度等参数。
            # 卡尔曼滤波可以根据Tracks状态预测下一帧的目标框状态
            tracker.predict()
            # 更新：预测值和观测值，两个正态分布的状态进行线性加权，得到目前系统预测的状态。
            # 卡尔曼滤波更新是对观测值(匹配上的Track)和估计值更新所有track的状态
            tracker.update(detections)
            indexIDs = []
            bboxes = []
            #dis_last = list(range(20))#[0]*20
            # 画yolo v3检测出的矩形框
            # for det in detections:
            #     bbox = det.to_tlbr()
            #     cv2.rectangle(frame0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            #跟踪相关
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # boxes.append([track[0], track[1], track[2], track[3]])
                # print('track.track_id', track.track_id)
                indexIDs.append(int(track.track_id))
                # print('indexIDs', indexIDs)
                counter.append(int(track.track_id))  # 这里将每一帧的id都放入，且不清除（因为初始化是在while外面），在下面用set函数进行去除相同元素
                # print('counter', counter)
                bbox = track.to_tlbr()
                bboxes.append(bbox)
                # print('bboxes', bboxes)
                #color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                color=(0,255,0)
                # 画deepsort 预测出的矩形框
                cv2.rectangle(frame0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 1)
                #frame0=put_mask(frame0, bbox)
                cv2.putText(frame0, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-3 * 150, (color), 1)
                if len(class_names) > 0:
                   class_name = class_names[0]
                   cv2.putText(frame0, str(class_names[0]),(int(bbox[0]), int(bbox[1]+20)),0, 5e-3 * 150, (color),1)

                i += 1
                # print(i )
                # 每个多目标的center[1]表示纵坐标
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                # print("运行到了这里 449")
                # track_id[center]
                pts[track.track_id].append(center)
                thickness = 1
                # center point
                cv2.circle(frame0, (center), 1, color, thickness)
                # print("运行到了这里 455")
                # draw motion path画运动轨迹
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame0, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
                # 测速测距模块函数 schling add
                #求相对图像坐标系速度
                pts_len=len(pts[track.track_id])
                if pts_len>1:
                    dist = ((pts[track.track_id][pts_len-1][0]- pts[track.track_id][pts_len-2][0]) ** 2 + (pts[track.track_id][pts_len-1][1] - pts[track.track_id][pts_len-2][1]) ** 2) ** 0.5
                    #print("track.track_id=%d pts_len=%d dist=%d"%(track.track_id, pts_len,dist))

                #求距离及是否碰撞需要碰撞预警
                z = mdm.mdm_info_proc(track.track_id, bbox, seconds,last_seconds,dis_last, fcw)

                z_0 = "{:.2f}".format(z)
                z_0=z_0+'m'
                cv2.putText(frame0, str(z_0), (int(bbox[0]+50), int(bbox[1] - 10)), 0, 5e-3 * 150, (color), 2)
            last_seconds = seconds

            FCW=1
            #出发FCW报警flag
            #if counter_jueduizhenhao%100==0 or counter_jueduizhenhao%100==1 or counter_jueduizhenhao%100==2 or counter_jueduizhenhao%100==3 or counter_jueduizhenhao%100==4  or counter_jueduizhenhao%100==6:

            # 清空模型时，对应表格的内容会同步清空
            self.tableWidget.clear()
            # print("运行到了这里 465")
            #print(len(indexIDs), len(class_names), len(bboxes))
            if len(indexIDs) > 0 and len(class_names) > 0 and len(bboxes) > 0:
                # table.write(counter_zhenhao - 2, 0, "第"+counter_jueduizhenhao+"帧")
                table.write(counter_zhenhao - 2, 0, "帧号：")
                table.write(counter_zhenhao - 2, 1, counter_jueduizhenhao)
                list_header = ['ID', 'Class', 'Point0', 'Point1', 'Point2', 'Point3']
                for j in range(len(list_header)):
                    table.write(counter_zhenhao - 1, j, list_header[j])

                # 往表格里写入ID
                for j in range(len(indexIDs)):
                    item1 = QTableWidgetItem(str(indexIDs[j]))
                    self.tableWidget.setItem(j, 0, item1)  # 第1列
                    table.write(counter_zhenhao + j, 0, indexIDs[j])#往表格里写数据

                # 往表格里写入类别
                for j in range(len(indexIDs)):
                    #print(j)
                    item2 = QTableWidgetItem(str(class_names[0]))
                    self.tableWidget.setItem(j, 1, item2)  # 第2列
                    table.write(counter_zhenhao + j, 1, class_names[0])
                # print("运行到了这里 477")
                # 往表格里写入坐标
                for j in range(len(indexIDs)):
                    # print(int(bboxes[j][0]), int(bboxes[j][1])), (int(bboxes[j][2]), int(bboxes[j][3]))
                    item3 = QTableWidgetItem(str((int)(bboxes[j][0])))
                    self.tableWidget.setItem(j, 2, item3)  # 第3列
                    item4 = QTableWidgetItem(str((int)(bboxes[j][1])))
                    self.tableWidget.setItem(j, 3, item4)  # 第3列
                    item5 = QTableWidgetItem(str((int)(bboxes[j][2])))
                    self.tableWidget.setItem(j, 4, item5)  # 第3列
                    item6 = QTableWidgetItem(str((int)(bboxes[j][3])))
                    self.tableWidget.setItem(j, 5, item6)  # 第3列
                    table.write(counter_zhenhao + j, 2, (int)(bboxes[j][0]))
                    table.write(counter_zhenhao + j, 3, (int)(bboxes[j][1]))
                    table.write(counter_zhenhao + j, 4, (int)(bboxes[j][2]))
                    table.write(counter_zhenhao + j, 5, (int)(bboxes[j][3]))

                counter_zhenhao += (len(indexIDs) + 3)
                #print(counter_zhenhao)

            # print("运行到了这里 489")
            count = len(set(counter))  # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
            if fcw[0] == 1:
                fcw[0] = 0  # 清除fcw报警标志
                bbox111 = [380, 200, 980, 300]
                #frame0 = put_mask(frame0, bbox111,0.5)
            #cv2.line(frame0, (0, 100), (200, 200), (0, 255, 255), 2)
            cv2.line(frame0, (0, (int)(height0/2)), (width0, (int)(height0/2)), (0, 255, 255), 2)
            #print("this 222")
            #车轮延长线
            cv2.line(frame0, ((int)(0.2 * width0), height0),
                     ((int)(0.2 * width0 + K0* height0 * 0.2), (int)(height0 * 0.8)),
                     (0, 255, 255), 2)

            #print("this 333")
            cv2.line(frame0, ((int)(0.8 * width0), height0),
                     ((int)(0.8 * width0 - K0 * height0 * 0.2), (int)(height0 * 0.8)),
                     (0, 255, 255), 2)
            #print("绝对帧号：")
            #print(counter_jueduizhenhao)
            fps = (fps + (1. / (time.time() - t1))) / 2
            cv2.putText(frame0, "FPS: %f" % (fps), (int(20), int(40)), 0, 1.2, (0, 255, 0), 2)
            cv2.putText(frame0, "Current Object Counter: " + str(i), (int(20), int(70)), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame0, "Total Object Counter: " + str(count), (int(20), int(100)), 0, 0.8, (0, 255, 0), 2)
            #cv2.putText(frame0, "jueduizhenhao: " +str(int(counter_jueduizhenhao)), (40, 130), 0, 0.8, (0, 255, 0), 2)
            if writeVideo_flag:
                # save a frame
                out.write(frame0)

        else:
            frame0 = self.frame.copy()
        # 在界面实时显示结果
        img = cv2.resize(frame0, (1280, 720), interpolation=cv2.INTER_AREA)
        height, width, bytesPerComponent = img.shape  # 取彩色图片的长、宽、通道
        bytesPerLine = 3 * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.label_ShowPicture.setPixmap(QPixmap(pixmap))
        file.save('data0904.xls')  # 保存txt文件


if __name__ == "__main__":
    # main.main(YOLO())

    yolo = YOLO()
    mdm = MDM()

    max_cosine_distance = 0.5  # 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3  # 非极大抑制的阈值
    counter = []  # 总目标数，得放总循环外面

    model_filename = 'model_data/market1501.pb'  #夹加载识别的模型
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    detections = []
    boxs = []
    # 画多目标跟踪的轨迹
    pts = [deque(maxlen=10) for _ in range(9999)]
    warnings.filterwarnings('ignore')
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3),
                               dtype="uint8")

    file = Workbook(encoding='utf-8')  # 指定file以utf-8的格式打开
    table = file.add_sheet('data0144')  # 指定打开的文件名，若没有，则新建一个
    counter_zhenhao = 3
    counter_jueduizhenhao = 0
    last_frame_num = 0

    writeVideo_flag = True
    # 写视频相关操作

    app = QtWidgets.QApplication(sys.argv)
    #os.system('python login_main.py')  #执行login_main.py文件，即登录界面，账号和密码为 diyun  12345
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

