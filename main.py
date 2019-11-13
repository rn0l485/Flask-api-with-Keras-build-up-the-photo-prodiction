import io

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None


def load_model():
    # load pre-trained 好的 Keras model，這邊使用 ResNet50 和 ImageNet 資料集（你也可以使用自己的 model）
    global model
    global graph
    model = ResNet50(weights='imagenet')
    # 初始化 tensorflow graph
    graph = tf.get_default_graph()


def preprocess_image(image, target):
    # 將圖片轉為 RGB 模式方便 predict
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 將資料進行前處理轉成 model 可以使用的 input
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {'success': False}
    print('request')
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('image'):
            # 從 flask request 中讀取圖片（byte str）
            image = request.files['image'].read()
            # 將圖片轉成 PIL 可以使用的格式
            image = Image.open(io.BytesIO(image))

            # 進行圖片前處理方便預測模型使用
            image = preprocess_image(image, target=(224, 224))

            # 原本初始化的 tensorflow graph 搭配 sesstion context，預測結果
            with graph.as_default():
                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
            
            data['predictions'] = []

            # 將預測結果整理後回傳 json 檔案（分類和可能機率）
            for (_, label, prob) in results[0]:
                r = {'label': label, 'probability': float(prob)}
                data['predictions'].append(r)

            data['success'] = True

    return jsonify(data)

# 當啟動 server 時先去預先 load model 每次 request 都要重新 load 造成效率低下且資源浪費。記得等到 model 和 server 完整執行後再發 request
if __name__ == '__main__':
    print(('* Loading Keras model and Flask starting server...'
        'please wait until server has fully started'))
    load_model()
    app.run()