# captcha_identification

[《使用 Keras 搭建模型识别四位数字字母验证码》](https://xiaosheng.run/2019/09/12/article167/)的代码示例

## 说明

参考图片分类任务中经典的 VGG 模型，采用 3 个块 (Block) 共计 6 个卷积层来识别 4 位数字字母验证码。

```python
pic_in = Input(shape=(32, 90, 1))
# Block 1
cnn_features = Conv2D(32, (3,3), activation='relu', padding='same')(pic_in)
cnn_features = Conv2D(32, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
# Block 2
cnn_features = Conv2D(64, (3,3), activation='relu',padding='same')(cnn_features)
cnn_features = Conv2D(64, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
# Block 3
cnn_features = Conv2D(128, (3,3), activation='relu',padding='same')(cnn_features)
cnn_features = Conv2D(128, (3,3), activation='relu')(cnn_features)
cnn_features = MaxPooling2D((2, 2))(cnn_features)
cnn_features = Dropout(0.25)(cnn_features)
cnn_features = Flatten()(cnn_features)
# classifier 1
output_l1 = Dense(128, activation='relu')(cnn_features)
output_l1 = Dropout(0.5)(output_l1)
output_l1 = Dense(36, activation='softmax')(output_l1)
# classifier 2
output_l2 = Dense(128, activation='relu')(cnn_features)
output_l2 = Dropout(0.5)(output_l2)
output_l2 = Dense(36, activation='softmax')(output_l2)
# classifier 3
output_l3 = Dense(128, activation='relu')(cnn_features)
output_l3 = Dropout(0.5)(output_l3)
output_l3 = Dense(36, activation='softmax')(output_l3)
# classifier 4
output_l4 = Dense(128, activation='relu')(cnn_features)
output_l4 = Dropout(0.5)(output_l4)
output_l4 = Dense(36, activation='softmax')(output_l4)

model = Model(inputs=pic_in, outputs=[output_l1, output_l2, output_l3, output_l4])
```

## 使用

下载最新训练好的 [model-45.hdf5](https://github.com/jsksxs360/captcha_identification/blob/master/model/model-45.hdf5) 模型（该模型在包含 4 万图片的测试集上取得 99.88% 的准确率）。

通过以下代码调用模型识别验证码：

```python
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
```

> **环境**
> 
> 测试环境为 Python3，Tensorflow 建议 1.14+，Keras 2.3.1。
> 
> 其他 Python 包依赖：scikit-image, numpy, flask, gevent, uuid 
