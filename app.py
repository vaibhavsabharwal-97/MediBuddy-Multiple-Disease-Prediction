import joblib
from flask import Flask, request, render_template, Response, jsonify

from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import pickle
import numpy as np

import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
from util import base64_to_pil
from flask import Flask, render_template, request


model = pickle.load(open('model.pkl', 'rb')) #Breast Cancer
model1 = pickle.load(open('xgboost_heart_disease_model.pkl', 'rb')) #Heart Disease

# Load the brain tumor model
brain_tumor_model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
bootstrap = Bootstrap(app)
model5 = 'pxray.h5' #chest xray

model6 = load_model(model5)

print('Model loaded. Start serving...')


def model_predict(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)

    return preds
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img):
    # Assuming `brain_tumor_model` is defined elsewhere
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = brain_tumor_model.predict(input_img)
    class_index = np.argmax(result, axis=-1)[0]
    return class_index

@app.route('/brain')
def brain():
    return render_template('brain.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_brain():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        class_index = getResult(file_path)
        result = get_className(class_index)
        return result
    return None
@app.route("/cancer") #Breast Cancer
def cancer():
    return render_template("cancer.html")

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))



@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    return render_template("liver.html")

def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('liver_disease_model.pkl')
        result = loaded_model.predict(to_predict)
        return result[0]  # Return the first (and only) element of the prediction array
    else:
        result = 0  # Assign a default value for cases where size != 7
        return result  # Return the integer value directly

@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePred(to_predict_list, len(to_predict_list))

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Liver Disease"
    return render_template("liver_result.html", prediction_text=prediction)


@app.route('/pneumonia', methods=['GET'])
def pneumonia():
    # Main page
    return render_template('pneumonia.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # Get the image from post request
        img = base64_to_pil(request.json)

        img.save("uploads\image.jpg")

        img_path = os.path.join(os.path.dirname(__file__), 'uploads\image.jpg')

        os.path.isfile(img_path)

        img = image.load_img(img_path, target_size=(64, 64))

        preds = model_predict(img, model6)

        result = preds[0, 0]

        print(result)

        if result > 0.5:
            return jsonify(result="PNEMONIA")
        else:
            return jsonify(result="NORMAL")

    return None

##################################################################################


#####################################################################

############################################################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "gender", "systolic blood pressure", "diastolic blood pressure", "cholesterol level",
                     "glucose level", "smoke", "alcohol intake", "active", "height_meter",
                     "weight_kg", "body mass index"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model1.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))

############################################################################################################


if __name__ == "__main__":
    app.run(debug=True)