# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>
from flask import Flask, jsonify, request
from fmodels import predict, get_models, batch_predict
#from service_streamer import ThreadedStreamer

app = Flask(__name__)
#streamer = ThreadedStreamer(stream_predict, batch_size=50176)

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        result = predict(img_bytes)
        return jsonify(result)

@app.route('/batch', methods=['POST'])
def batch():
    if request.method == 'POST':
        # get the list of files and read them as bytes
        files = request.files.getlist("file")
        image_bytes = [f.read() for f in files]
        # feed to batch predict method
        result = batch_predict(image_bytes)
        return jsonify(result)

@app.route('/models', methods=['GET'])
def models():
    if request.method == 'GET':
        result = get_models()
        return jsonify(result)

# # experimental utilize streamer library for multi threaded predict
# @app.route('/sclass', methods=['POST'])
# def stream_classify():
#     if request.method == 'POST':
#         file = request.files['file']
#         img_bytes = file.read()
#         result = streamer.predict(img_bytes)
#         return jsonify(result)




if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)