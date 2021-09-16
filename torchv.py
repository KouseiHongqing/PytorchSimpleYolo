'''
函数说明: 这个和老师的不太一样，因为老师模型加载不进去 我就自己撸了个差不多的 用的yolov4 建议用yolov5 集成度更高更方便
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 16:49:25
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
import scipy.misc
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
# from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_b
%matplotlib inline
##官方给的NMS
# from torchvision.ops import nms
#自己手撸pytorch版 可以学习一下
from pytorchnms import NMS
#pytorch 的summary 要去github下载torchsummary https://github.com/sksq96/pytorch-summary
#from torchsummary import summary
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
from models import *
np.random.seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,dataset
box_confidence =torch.randn(19,19,5,1)
boxes = torch.randn(19,19,5,4)
box_class_probs = torch.randn(19,19,5,80)
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores = torch.multiply(box_confidence, box_class_probs)
    box_classes = torch.argmax(box_scores, axis=-1) # 获取概率最大的那个种类的索引
    box_class_scores = torch.max(box_scores, axis=-1)[0]
    filtering_mask = torch.greater_equal(box_class_scores, threshold)
    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]
    return scores, boxes, classes
scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))


#没写训练过程 这里直接训练好的飞机模型
#测试图片 和是否用GPU
def predict(imgfile="./images/test.jpg",use_cuda = False):
    import sys
    import cv2
    n_classes=2 #只有2个分类 oiltank 和 plane
    weightfile="./data/yolov4.conv.137.pth" 
    namesfile = './data/coco.names'
    width=416
    height=416
    model = Yolov4(yolov4conv137weight=weightfile, n_classes=n_classes, inference=True)
    # pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    # model.load_state_dict(pretrained_dict)
    if use_cuda:
        model.cuda()
    img = cv2.imread(imgfile)

    # 推理输入是416
    # 训练输入可以是608或者任意值
    #  可选输入
    #   Hight {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    for i in range(2):# for循环是为了快速检查，因为第一次循环一般耗时较长
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    resultImg=plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
    cv2.imshow("image",resultImg)
    cv2.waitKey(0)

predict("./data/aircraft_4.jpg")