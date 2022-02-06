# Waste Image Classification Website

<br>

The front page of the website:
<p align="center">
  <img width="1440" alt="Screenshot 2022-01-19 at 1 47 49 PM" src="https://user-images.githubusercontent.com/46462603/150212757-acd9b457-bf01-4077-a9b5-d800ae9ef95c.png">
</p>

<br>

The website after the user uploaded a photo of straws. The output 'The item shown is recyclable' can be seen at the bottom of the uploaded photo.
<p align="center">
<img width="1440" alt="Screenshot 2022-01-19 at 1 57 29 PM" src="https://user-images.githubusercontent.com/46462603/150212763-69406669-001e-42ff-add6-0fa96d5c1805.png">
</p>

<br>

A sample of the dataset. 
<p align="center">
  <img width="592" alt="Screenshot 2022-01-19 at 2 01 51 AM" src="https://user-images.githubusercontent.com/46462603/150097996-8bae695e-ac00-4cd2-bda6-0524bfd1b7ef.png">
</p>

<br>

## Goal
To create a website that helps the user classify images of wastes into organic and recycling wastes. 

## Dataset
The dataset is obtained from <a href="https://www.kaggle.com/techsash/waste-classification-data" target="_blank">Waste Classification data on Kaggle</a>. There are altogether 22564 images in the training folder and 2513 images in the test folder. I split some of the photos randomly to create another folder named as the validation set. In my project, there are 19998 images in the training set, 475 images in the validation set, and 2038 images in the test set. 

In each of the training, validation, and test folders, there are 2 subfolders named 'O' and 'R'. The subfolders 'O' contain images of organic items such as fruits and vegetables while the subfolders 'R' contain images of recyclable items such as plastic bottles and papers. Some of the images have clear, white backgrounds while some of them have more complex backgrounds. Some examples of the dataset can be found at <a href="https://www.kaggle.com/techsash/waste-classification-data" target="_blank">Waste Classification data on Kaggle</a> or the <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Waste_Image_Classification.ipynb" target="_blank">Waste_Image_Classification.ipynb</a> file.

## Code, files, or folders needed to run the program
- <a href="https://github.com/ZhengEnThan/Waste-Classification/tree/main/templates" target="_blank">templates</a>
- <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Procfile" target="_blank">Procfile</a>
- <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Waste_Image_Classification.ipynb" target="_blank">Waste_Image_Classification.ipynb</a>
- <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/requirements.txt" target="_blank">requirements.txt</a>
- <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/setup.sh" target="_blank">setup.sh</a>
- <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/waste_streamlit_app.py" target="_blank">waste_streamlit_app.py</a>
- waste_model.h5, which can be downloaded by running the last cell of <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Waste_Image_Classification.ipynb" target="_blank">Waste_Image_Classification.ipynb</a>

## How to use the program
If you are running the code locally, you will not have to connect to your Google Drive as what I have done in <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Waste_Image_Classification.ipynb" target="_blank">Waste_Image_Classification.ipynb</a>. In general, you would have to edit the code cells where directory paths of the dataset and files are defined so that they match the directories where you put these files in. Therefore, the code cells that you might have to edit are the second, third and last code cells in <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Waste_Image_Classification.ipynb" target="_blank">Waste_Image_Classification.ipynb</a>. 

After making sure that all the files or folders needed to run the program are in the same directory, simply enter the command 'streamlit run waste_streamlit_app.py' in the terminal to run the web application. The website will automatically pop up in the browser. You may then upload a photo of anything, such as a broccoli, and see what the website tells you about the item in the photo you uploaded. The output text will be visible at the bottom of the uploaded photo.

## What have I learned
- Used tensorflow.keras to load images from directories.
- Used pandas to create a dataframe containing the images as data and matplotlib to display the data.
- Used tensorflow.keras to create an instance of InceptionV3 with the pre-learned imagenet weights and added another 2 dense layers to complete the image classification model.
- Used EarlyStopping from tensorflow.keras in fitting the model to the training set.
- Used sklearn.metrics to evaluate the model's performance on the test set using confusion_matrix and ConfusionMatrixDisplay, classification_report, and RocCurveDisplay.
- Saved the model into a H5 file.

## Main libraries or modules used
- tensorflow.keras
- sklearn.metrics
- numpy
- seaborn
- matplotlib
- pandas

## Approaches
The InceptionV3 model with Imagenet weights is used. All the layers except the last one are frozen to avoid destroying any of the information they contain in future training. The last layer of the neural network is added and trained with our training images. Since we want to use probabilities as a basis of how we decide whether the item in a given image is organic or recyclable, the activation function used for the last layer is the Sigmoid function. 

Using EarlyStopping with a patience of 5 epochs on validation loss, the model fit the training data with a validation loss of 0.2896 and validation accuracy of 0.8737 or 87%. The trained model is able to achieve an accuracy of around 83% when tested on the test set. 

Streamlit is used to create and run the web application.

## Comments
The file waste_model.h5 used in <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/waste_streamlit_app.py" target="_blank">waste_streamlit_app.py</a> can be downloaded in the <a href="https://github.com/ZhengEnThan/Waste-Classification/blob/main/Waste_Image_Classification.ipynb" target="_blank">Waste_Image_Classification.ipynb</a> file. The waste_model.h5 file is not uploaded onto GitHub because its size is too large.

From what I have observed, the trained waste image classification model tends to associate bright and colourful items with organic wastes. Therefore, it would sometimes classify colourful straws as organic wastes.  

## References
- Image Augmentation on the fly using Keras ImageDataGenerator <br>
  https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
- What exactly does shear do in ImageDataGenerator of Keras <br>
  https://stackoverflow.com/questions/57301330/what-exactly-the-shear-do-in-imagedatagenerator-of-keras
- Why we have to rescale by 1. / 255 <br>
  https://github.com/Arsey/keras-transfer-learning-for-oxford102/issues/1
- Tutorial on using Keras flow_from_directory and generators <br>
  https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
- Transfer learning and fine-tuning <br>
  https://www.tensorflow.org/guide/keras/transfer_learning
- Deep Learning A-Z™: Convolutional Neural Networks (CNN) - Step 3: Flattening <br>
  https://www.slideshare.net/KirillEremenko/deep-learning-az-convolutional-neural-networks-cnn-step-3-flattening
- Module: tf.keras.models <br>
  https://www.tensorflow.org/api_docs/python/tf/keras/models
- Dense layer binary classification cannot be set to 2 <br>
  https://stackoverflow.com/questions/67127120/dense-layer-binary-classification-cannot-be-set-to-2

~ Project created in January 2022 ~
