from flask import Flask, url_for
from flask import request
from flask import json
from skimage import io
from skimage import color
import numpy as np
import os
import re
from keras.models import Model
from keras.models import load_model
import base64
import uuid
import gevent
from gevent.pywsgi import WSGIServer
from gevent import monkey
monkey.patch_all()

app = Flask(__name__)

model = None

def loading():
    global model
    model = load_model('../model/model-45.hdf5')

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

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/captcha', methods = ['POST'])
def api_message():
    base64_str = ''
    if request.headers['Content-Type'] == 'text/plain':
        base64_str = request.data
    elif request.headers['Content-Type'] == 'application/json':
        base64_str = request.json.get('pic', '')
    else:
        return json.dumps({"status":"NO", "msg":"request head not support"})
    if base64_str:
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", base64_str, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            data = result.groupdict().get("data")
        else:
            data = src
        try:
            imgdata = base64.b64decode(data)
            tempfilename = str(uuid.uuid1()) + '.png'
            file = open(tempfilename,'wb')
            file.write(imgdata)
            file.close()
            im = io.imread(tempfilename)
            im3 = color.rgb2gray(im)
            pic = np.asarray([np.expand_dims(im3, axis=2)])
            result = np.squeeze(model.predict(pic))
            os.remove(tempfilename)
            return json.dumps({"status":"OK", "result": decode(result)})
        except:
            return json.dumps({"status":"NO", "msg":"error"})
    else:
        return json.dumps({"status":"NO", "msg":"input base64 is empty"})

if __name__ == '__main__':
    loading() # 加载模型
    http_server = WSGIServer(('0.0.0.0', 8090), app)
    http_server.serve_forever()
