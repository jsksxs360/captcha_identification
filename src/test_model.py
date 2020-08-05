#coding:utf-8
from skimage import io
from skimage import color
import numpy as np
import os
from keras.models import Model
from keras.models import load_model

# 构建映射字典
labels = ['a','b','c','d','e','f','g',
          'h','i','j','k','l','m','n',
          'u','p','q','r','s','t','o',
          'v','w','x','y','z','0','1',
          '2','3','4','5','6','7','8','9']
label2id = {}
id2label = {}

for label in labels:
    label2id[label] = len(label2id)
for label, id in label2id.items():
    id2label[id] = label

def decode(y):
    y = np.argmax(np.array(y), axis=-1)
    return ''.join([id2label[x] for x in y])

im = io.imread('1a8m.png')
im3 = color.rgb2gray(im)
pic = np.asarray([np.expand_dims(im3, axis=2)])
model = load_model('../model/model-45.hdf5')
py = np.squeeze(model.predict(pic))
print(decode(py))