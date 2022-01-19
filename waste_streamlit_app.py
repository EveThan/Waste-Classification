import tensorflow as tf
model = tf.keras.models.load_model('waste_model.h5')

import streamlit as st
st.write("""
         # Waste Classification
         """
         )
st.write("This is a simple image classification web app to classify wastes")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

#import cv2
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img , img_to_array

def import_and_predict(image_data, model):
    size = (150,150)
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = img_to_array(img)
    img = img.reshape(1, 150 ,150 ,3)
    img = img.astype('float32')
    img = img/255.0
    
    prediction = model.predict(img)
        
    return prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    labels = np.array(prediction)
    labels[labels > 0.5] = 1
    labels[labels <= 0.5] = 0
    
    final = np.array(labels)
    
    if final[0][0] == 0:
        st.write("The item shown is organic")
    else:
        st.write("The item shown is recyclable")