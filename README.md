# Convolutional-Neural-Network

## Description
This project features, data & feature engineering, data preprocessing, and a convolutional neural network. 
Credit to the book Neural Network Projects with Python by James Loy. The file main.py contains all the functions
in the order executed. The data engineering and model structure can be viewed in main.py. The CNN detects whether
a given image is a cat or dog. 

## Installation
* Pip install tensorflow (built with 2.5.0)
* Pip install numpy (built with 1.19.5)
* Pip install piexif (built with 1.1.3)

## Usage
Running the main.py will load the data, engineer the features in such data, preprocess the data and then 
train the model on the data. 

To get better usage out of this project the functions should be read and understood in the order executed. 
Alongside the book Neural Network Projects with Python by James Loy. 

## Neural Network Details
Model: "CNN"  
Layer (type)  |               Output Shape          |    Param #   
conv2d (Conv2D)              (None, 30, 30, 32)        896       
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
flatten (Flatten)            (None, 1568)              0         
dense (Dense)                (None, 128)               200832    
dropout (Dropout)            (None, 128)               0         
dense_1 (Dense)              (None, 1)                 129       
Total params: 201,857  
Trainable params: 201,857  
Non-trainable params: 0  

Testing Accuracy: 77.8% 

## Credits
* Author: James Loy
* Modified & Studied by: Lee Taylor

