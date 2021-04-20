import torch
from torch.autograd import Variable
import utils
import mydataset
from PIL import Image
import crnn as crnn
import torch.nn.functional as F
import keys
import config
import os
from mydataset import data_tf_fullimg

alphabet = keys.alphabet_v2
converter = utils.strLabelConverter(alphabet.copy())


def val_model(txt_folder, img_folder, model, gpu, log_file = '0625.log'):
    h = open('log/{}'.format(log_file),'w')
    TXTs = os.listdir(txt_folder)
    num_all = 0
    num_correct = 0
    for TXT in TXTs:
        fname = os.path.join(img_folder, TXT.replace('.txt', '.jpg'))
        TXT_file = os.path.join(txt_folder, TXT)
        with open(TXT_file) as f:
            content = f.readlines()
            for line in content:
                w1, h1, w2, h2, w3, h3, w4, h4, script, label = line.strip().split(',', 9)
                if set(label) == set('#'):
                    continue
                left = min(int(w1), int(w2), int(w3), int(w4))
                right = max(int(w1), int(w2), int(w3), int(w4))
                top = min(int(h1), int(h2), int(h3), int(h4))
                bottom = max(int(h1), int(h2), int(h3), int(h4))
                loc = [left, top, right, bottom]
                img = Image.open(fname)
                img = data_tf_fullimg(img, loc)
                img = img.convert('L')
                res = val_on_image(img,model,gpu)
                res = res.strip()
                label = label.strip()
                if res == label:
                    num_correct+=1
                else:
                    print('filename:{}\npred  :{}\ntarget:{}'.format(fname, res, label))
                    h.write('filename:{}\npred  :{}\ntarget:{}\n'.format(fname,res, label))
                num_all+=1
    h.write('ocr_correct: {}/{}/{}\n'.format(num_correct,num_all,num_correct/num_all))
    print(num_correct/num_all)
    h.close()
    return num_correct, num_all

def val_on_image(img,model,gpu):
    imgH = 32
    W, H = img.size[:2]
    imgW = W * H // imgH
    imgW = max(imgW, 5) # W至少是5

    transformer = mydataset.resizeNormalize((imgW, imgH))
    image = transformer(img)
    if gpu:
        image = image.cuda()
    image = image.view(1, *image.size()) # 三维变四维，相当于batchsize = 1
    image = Variable( image )

    model.eval()
    preds = model( image )
    preds = F.log_softmax(preds,2)
    conf, preds = preds.max( 2 )
    preds = preds.transpose( 1, 0 ).contiguous().view( -1 )
    preds_size = Variable( torch.IntTensor( [preds.size( 0 )] ) )
    # raw_pred = converter.decode( preds.data, preds_size.data, raw=True )
    sim_pred = converter.decode( preds.data, preds_size.data, raw=False )
    return sim_pred


if __name__ == '__main__':
    import sys
    model_path = './crnn_models/CRNN-0627-crop_48_901.pth'
    gpu = True
    if not torch.cuda.is_available():
        gpu = False

    model = crnn.CRNN(config.imgH, 1, len(alphabet) + 1, 256)
    if gpu:
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    if gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if len(sys.argv)>1 and 'train' in sys.argv[1]:
        infofile = 'data_set/infofile_updated_0627_train.txt'
        print(val_model(infofile, model, gpu, '0627_train.log'))
    elif len(sys.argv)>1 and 'gen' in sys.argv[1]:
        infofile = 'data_set/infofile_0627_gen_test.txt'
        print(val_model(infofile, model, gpu, '0627_gen.log'))
    else:
        infofile = 'data_set/infofile_updated_0627_test.txt'
        print(val_model(infofile, model, gpu, '0627_test.log'))




