import os
import flask
import base64
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img , img_to_array

UPLOAD_FOLDER = os.path.join('static', 'waste_image')
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the classification model
model = load_model('waste_model.h5', compile = True)

@app.route('/')
def home():
    # Load the html file as layout
    return render_template("waste.html")

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        # Obtain the file uploaded by the user
        file = request.files['file']
        filename = file.filename
        
        # Make sure that the user has uploaded an image
        if (filename.rsplit('.', 1)[1] not in ALLOWED_EXT):
            error_message = "Please upload the correct file type. Only images can be accepted."
            return render_template("waste.html", waste_type = error_message)
        
        image_uploaded = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_uploaded)
        
        # Load and adjust the image 
        img = load_img(image_uploaded , target_size = (150 , 150))
        img = img_to_array(img)
        img = img.reshape(1, 150 ,150 ,3)
        img = img.astype('float32')
        img = img/255.0
    
        # Let the model predict the output given the image
        result = model.predict(img)
        
        labels = np.array(result)
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0
    
        final = np.array(labels)
    
        if final[0][0] == 0:
            waste_type = "Organic"
        else:
            waste_type = "Recyclable"
    
    return render_template("waste.html", image_uploaded = image_uploaded, waste_type = waste_type)
        
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 8080)