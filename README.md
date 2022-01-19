<p align="center">
  <img width="1440" alt="Screenshot 2022-01-19 at 1 47 49 PM" src="https://user-images.githubusercontent.com/46462603/150211930-d08fa908-03c4-4da0-afc5-da9d50b3fa99.png">
<img width="1440" alt="Screenshot 2022-01-19 at 1 49 36 PM" src="https://user-images.githubusercontent.com/46462603/150211938-e3900ba0-b816-4e6e-9276-9c7081b2c7da.png">
  <img width="592" alt="Screenshot 2022-01-19 at 2 01 51 AM" src="https://user-images.githubusercontent.com/46462603/150097996-8bae695e-ac00-4cd2-bda6-0524bfd1b7ef.png">
</p>

## Goal
To create a machine learning model that is able to differentiate between organic and recylcing wastes based on their images.

## Dataset
The dataset is obtained from https://www.kaggle.com/techsash/waste-classification-data. There are altogether 22564 images in the training folder and 2513 images in the test folder. I put split some of the photos randomly to create another folder named as the validation set. In my project, there are 19998 images in the training set, 475 images in the validation set, and 2038 images in the test set. 

## Approaches
The InceptionV3 model with Imagenet weights is used. The last layer of the neural network is trained again with our images. The trained model is able to achieve an accuracy of around 83% when tested on the test set. 

## Comments
The model tends to associate bright and colourful items with organic wastes. Therefore, it would sometimes classify colourful straws as organic wastes.

## References
https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
https://stackoverflow.com/questions/57301330/what-exactly-the-shear-do-in-imagedatagenerator-of-keras
https://github.com/Arsey/keras-transfer-learning-for-oxford102/issues/1
https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
https://www.tensorflow.org/guide/keras/transfer_learning
https://www.slideshare.net/KirillEremenko/deep-learning-az-convolutional-neural-networks-cnn-step-3-flattening
https://www.tensorflow.org/api_docs/python/tf/keras/models
https://stackoverflow.com/questions/67127120/dense-layer-binary-classification-cannot-be-set-to-2



