import os
import numpy as np
from flask import Flask, request, render_template, jsonify,redirect,url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import requests
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

model = tf.keras.models.load_model('my_mmodel.h5')
r=[]
with open("labels.txt","r") as file:
    r=file.readlines()

def predict_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    print("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(r[np.argmax(score)], 100 * np.max(score)))
    return r[np.argmax(score)]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file extension'})
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded image and make predictions
        prediction = [predict_image(file_path)]

        return render_template('ret.html', data=prediction)

    return render_template('index.html')
@app.route('/try', methods=['GET', 'POST'])
def tryh():
    return render_template("try.html")


@app.route('/ret')
def ret():
    message = request.args.get('message', '')
    section = request.args.get('section', '')
    return render_template('msent.html', message=message, section=section)


if __name__ == '__main__':
    app.run(debug=True)
