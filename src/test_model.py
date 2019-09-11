#coding:utf-8
from tqdm import tqdm
import scipy
from scipy import misc
import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Model
from keras.models import load_model

data_dir = '../data/cap_data/'
model_dir = '../model/'

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

dev_data = []
dev_labels1 = []
dev_labels2 = []
dev_labels3 = []
dev_labels4 = []
i = 0
for file_name in tqdm(sorted(os.listdir(data_dir))):
    if not file_name.endswith('.png'):
        continue
    if i % 10 == 0:
        dev_data.append(np.expand_dims(misc.imread(os.path.join(data_dir, file_name), mode='L'), axis=2))
        l1, l2, l3, l4 = tuple(str(file_name)[:4])
        dev_labels1.append(label2id[l1])
        dev_labels2.append(label2id[l2])
        dev_labels3.append(label2id[l3])
        dev_labels4.append(label2id[l4])
    i += 1

# 转换为 numpy 对象
dev_data = np.asarray(dev_data)
dev_labels1 = to_categorical(np.asarray(dev_labels1), num_classes=36)
dev_labels2 = to_categorical(np.asarray(dev_labels2), num_classes=36)
dev_labels3 = to_categorical(np.asarray(dev_labels3), num_classes=36)
dev_labels4 = to_categorical(np.asarray(dev_labels4), num_classes=36)

def decode(y):
    y = np.argmax(np.array(y), axis=-1)
    return ''.join([id2label[x] for x in y])

def test(model_name):
    cy_list = []
    py_list = []
    print('MODEL {} TESTING...'.format(model_name))
    model = load_model(os.path.join(model_dir, model_name))
    for i in range(len(dev_data)):
        X = np.asarray([dev_data[i]])
        y = [dev_labels1[i], dev_labels2[i], dev_labels3[i], dev_labels4[i]]
        py = np.squeeze(model.predict(X))
        cy_list.append(decode(y))
        py_list.append(decode(py))
    current_num = [1 if cy == py else 0 for cy, py in zip(cy_list, py_list)]
    print('MODEL: {}\tACC: {}'.format(model_name, sum(current_num)/len(cy_list)))

test('model-40.hdf5')
test('model-41.hdf5')
test('model-47.hdf5')
test('model-50.hdf5')