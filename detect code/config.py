import os

# base_dir = 'path to dataset base dir'
base_dir = './images'
img_dir = os.path.join(base_dir, 'VOC2007_text_detection/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007_text_detection/Annotations')

icdar17_mlt_img_dir = './train_data/train_img/'
icdar17_mlt_gt_dir = './train_data/train_label/'
num_workers = 0
# pretrained_weights = 'checkpoints/v3_ctpn_ep22_0.3801_0.0971_0.4773.pth'
pretrained_weights = 'E:/nndl-final-pj/other\'s code/OCR-task/OCR-code(CTPN+CRNN+CTC Loss)/checkpoints/CTPN.pth'


anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

checkpoints_dir = 'E:/nndl-final-pj/other\'s code/OCR-task/OCR-code(CTPN+CRNN+CTC Loss)/checkpoints'
outputs = r'./logs'
