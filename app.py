from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os


app = Flask(__name__)


MODEL_ARCHITECTURE = 'models\model_pathology_covid.json'
MODEL_WEIGHTS = 'models\model_pathology_covid_weights.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


# Get weights into the model
model=load_model(MODEL_WEIGHTS)
# print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    xtest_image = image.load_img(img_path, target_size=(224, 224))
    xtest_image = image.img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = model.predict_classes(xtest_image)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/index', methods=['GET'])
def AfterRegister():
    # Main page
    return render_template('index.html')

@app.route("/login")
def login():
  return render_template("login.html")

@app.route("/register")
def register():
  return render_template("register.html")

@app.route("/form")
def form():
  return render_template("form.html")

@app.route("/covid")
def covid():
  return render_template("detection.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        
        if preds[0][0] == 0:
            prediction = 'Positive For Covid-19'
        else:
            prediction = 'Negative for Covid-19'
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1)               # Convert to string
        return prediction
    return None

@app.route("/pneumonia")
def pneumonia():
  return render_template("detect.html")

@app.route('/prediction', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        if preds[0][0] == 0:
            prediction = 'Infected with Pneumonia'
        else:
            prediction = 'Not infected'
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1)               # Convert to string
        return prediction
    return None

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='192.168.0.106')

