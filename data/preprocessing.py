#coding:utf-8

from tqdm import tqdm
import pandas as pd
import os
import shutil
import scipy
from scipy import misc

pic_src_dir = 'cap_img/'
pic_result_dir = 'cap_data/'

if not os.path.exists(pic_result_dir):
    os.mkdir(pic_result_dir)

pic_info = pd.read_excel('qr.xlsx', header=None)

pic2cap = {} # 图片转验证码字典

for file_name, cap in zip(pic_info[0], pic_info[1]):
    pic2cap[file_name] = str(cap).lower()

print('preprocessing...')
for file_name in tqdm(sorted(os.listdir(pic_src_dir))):
    if not file_name.endswith('.png'):
        continue
    if file_name in pic2cap:
        if len(pic2cap[file_name]) != 4: # 过滤错误验证码
            continue
        try:
            misc.imread(os.path.join(pic_src_dir, file_name), mode='L')
        except:
            print('{} image damaged'.format(file_name))
            continue
        try:
            shutil.copy(os.path.join(pic_src_dir, file_name), os.path.join(pic_result_dir, pic2cap[file_name] + '.png'))
        except:
            print('copying {} failed'.format(file_name))
            pass
        