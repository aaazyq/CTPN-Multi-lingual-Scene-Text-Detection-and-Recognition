import pickle as pkl
import os
'''生成字母表'''
# =============================================================================
# # gen alphabet via label
# path = 'train_data/gt/'
# alphabet_set = set()
# infofiles = os.listdir(path)
# ind = 0
# for infofile in infofiles:
#     infofile = os.path.join(path, infofile)
#     f = open(infofile)
#     content = f.readlines()
#     f.close()
#     for line in content:
#         top1, left1, bottom1, left2, bottom2, right1, top2, right2, script, label = line.strip().split(',', 9)
#         for ch in label:
#             alphabet_set.add(ch)
# 
# alphabet_list = sorted(list(alphabet_set))
# pkl.dump(alphabet_list,open('alphabet.pkl','wb'))
# =============================================================================

'''调用保存好的字母表'''
alphabet_list = pkl.load(open('alphabet.pkl','rb'))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
# print(alphabet_v2)