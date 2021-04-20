import keys
# 参数设置

train_img_folder = './train_data/img/'
train_txt_folder = './train_data/gt/'
test_img_folder = './test_data/img/'
test_txt_folder = './test_data/gt/'
# =============================================================================
# # 训练集结果路径设定，数据格式为路径\t内容
# train_infofile = 'data_set/infofile_train_10w.txt'
# # 图像文件夹
# train_infofile_fullimg = ''
# # 测试集结果路径设定，数据格式为路径\t内容
# val_infofile = 'data_set/infofile_test.txt'
# =============================================================================
# 字母表
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
# 超参
workers = 0
batchSize = 64
niter = 40
lr = 0.00005
beta1 = 0.5
# 标准处理后图片尺寸
imgH = 32
imgW = 280
# 
nc = 1
nclass = len(alphabet)+1
nh = 256 # 隐层数目
# CPU与GPU
cuda = True
ngpu = 1
# 模型处理
pretrained_model = ''
saved_model_dir = 'crnn_models'
saved_model_prefix = 'CRNN'
use_log = False
remove_blank = False

displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = True
adadelta = False
keep_ratio = True
random_sample = True

