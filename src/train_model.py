#coding:utf-8
from tqdm import tqdm
from skimage import io
from skimage import color
import numpy as np
import os
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

data_dir = '../data/cap_data/'
model_dir = '../model/'
epochs = 50
batch_size = 256

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# 构建映射字典
labels = ['a','b','c','d','e','f','g',
          'h','i','j','k','l','m','n',
          'u','p','q','r','s','t','o',
          'v','w','x','y','z','0','1',
          '2','3','4','5','6','7','8','9']
label2id = {} # 标签转id
id2label = {} # id转标签

for label in labels:
    label2id[label] = len(label2id)
for label, id in label2id.items():
    id2label[id] = label

# 数据集
pic_data = []
labels1 = []
labels2 = []
labels3 = []
labels4 = []
print('PREPARING...')
for file_name in tqdm(sorted(os.listdir(data_dir))):
    if not file_name.endswith('.png'):
        continue
    im = io.imread(os.path.join(data_dir, file_name))
    im3 = color.rgb2gray(im)
    pic_data.append(np.expand_dims(im3, axis=2))
    l1, l2, l3, l4 = tuple(str(file_name)[:4]) # 切分出验证码的四位标签
    labels1.append(label2id[l1.lower()])
    labels2.append(label2id[l2.lower()])
    labels3.append(label2id[l3.lower()])
    labels4.append(label2id[l4.lower()])

# 按9:1划分训练集和验证集
train_data = [j for i, j in enumerate(pic_data) if i % 10 != 0]
train_labels1 = [j for i, j in enumerate(labels1) if i % 10 != 0]
train_labels2 = [j for i, j in enumerate(labels2) if i % 10 != 0]
train_labels3 = [j for i, j in enumerate(labels3) if i % 10 != 0]
train_labels4 = [j for i, j in enumerate(labels4) if i % 10 != 0]
dev_data = [j for i, j in enumerate(pic_data) if i % 10 == 0]
dev_labels1 = [j for i, j in enumerate(labels1) if i % 10 == 0]
dev_labels2 = [j for i, j in enumerate(labels2) if i % 10 == 0]
dev_labels3 = [j for i, j in enumerate(labels3) if i % 10 == 0]
dev_labels4 = [j for i, j in enumerate(labels4) if i % 10 == 0]

# 转换为 numpy 对象
train_data = np.asarray(train_data)
train_labels1 = to_categorical(np.asarray(train_labels1), num_classes=36)
train_labels2 = to_categorical(np.asarray(train_labels2), num_classes=36)
train_labels3 = to_categorical(np.asarray(train_labels3), num_classes=36)
train_labels4 = to_categorical(np.asarray(train_labels4), num_classes=36)
dev_data = np.asarray(dev_data)
dev_labels1 = to_categorical(np.asarray(dev_labels1), num_classes=36)
dev_labels2 = to_categorical(np.asarray(dev_labels2), num_classes=36)
dev_labels3 = to_categorical(np.asarray(dev_labels3), num_classes=36)
dev_labels4 = to_categorical(np.asarray(dev_labels4), num_classes=36)

# 构建模型
pic_in = Input(shape=(32, 90, 1))
cnn_features = Conv2D(64, (3,3), activation='relu', padding='same')(pic_in)
cnn_features = Conv2D(64, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
cnn_features = Conv2D(128, (3,3), activation='relu',padding='same')(cnn_features)
cnn_features = Conv2D(128, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
cnn_features = Conv2D(256, (3,3), activation='relu',padding='same')(cnn_features)
cnn_features = Conv2D(256, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
cnn_features = Flatten()(cnn_features)
output_l1 = Dense(256, activation='relu')(cnn_features)
output_l1 = Dropout(0.5)(output_l1)
output_l1 = Dense(36, activation='softmax')(output_l1)
output_l2 = Dense(256, activation='relu')(cnn_features)
output_l2 = Dropout(0.5)(output_l2)
output_l2 = Dense(36, activation='softmax')(output_l2)
output_l3 = Dense(256, activation='relu')(cnn_features)
output_l3 = Dropout(0.5)(output_l3)
output_l3 = Dense(36, activation='softmax')(output_l3)
output_l4 = Dense(256, activation='relu')(cnn_features)
output_l4 = Dropout(0.5)(output_l4)
output_l4 = Dense(36, activation='softmax')(output_l4)

model = Model(inputs=pic_in, outputs=[output_l1, output_l2, output_l3, output_l4])
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              loss_weights=[1., 1., 1., 1.], # 给四个分类器同样的权重
              metrics=['accuracy'])

def decode(y):
    y = np.argmax(np.array(y), axis=-1)
    return ''.join([id2label[x] for x in y])

class Evaluate(Callback):
    def __init__(self):
        self.epoch = 1
        self.cy_list = []
        for i in range(len(dev_data)):
            y = [dev_labels1[i], dev_labels2[i], dev_labels3[i], dev_labels4[i]]
            self.cy_list.append(decode(y))
        self.dev_size = len(self.cy_list)

    def on_epoch_end(self, epoch, logs=None):
        py_list = []
        for i in range(len(dev_data)):
            X = np.asarray([dev_data[i]])
            py = np.squeeze(model.predict(X))
            py_list.append(decode(py))
        with open('evaluate.txt', 'at') as writer:
            current_num = [1 if cy == py else 0 for cy, py in zip(self.cy_list, py_list)]
            writer.write('EPOCH' + str(self.epoch) + '\n')
            writer.write('MODEL ACC:' + str(sum(current_num)/self.dev_size) + '\n\n')
        self.epoch += 1

if __name__ == '__main__':
    evaluator = Evaluate()
    checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, 'model-{epoch:02d}.hdf5'), 
                                    verbose=1)
    model.fit(train_data, [train_labels1, train_labels2, train_labels3, train_labels4],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(dev_data, [dev_labels1, dev_labels2, dev_labels3, dev_labels4]),
            callbacks=[checkpointer, evaluator])
    
