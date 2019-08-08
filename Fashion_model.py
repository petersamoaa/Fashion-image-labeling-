"""
Created on Tue Aug  6 22:47:55 2019

@author: Peter Samoaa
"""
# Import the data
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
# import the dataset 
fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep= ',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep= ',')
# explose the data 
# to explore the data regarding the images we  want to define our dataset as matrix 
training = np.array(fashion_train_df,dtype = "float32")
testing = np.array(fashion_test_df,dtype = "float32")
# visulaize the image 
# view the imgaes in grid format 
# define the dimentions of grid 
W_grid = 15
L_grid = 15
# we use subplot to return the figure object and axes object
fig, axes = plt.subplots(L_grid, W_grid, figsize = (18,18))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_training = len(training) # get the length of the training dataset
# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 10)
    axes[i].axis('off') # avoid showing the number of pixels around the images 

plt.subplots_adjust(hspace=0.8)# adjust the space just to height between the images

# Remember the 10 classes decoding is as follows:
# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot
        
# prepare the X & y
X_train = training[:,1:]/225
y_train = training[:,0]

X_test = testing[:,1:]/225
y_test = testing[:,0]

# define the training and validation data 
# validation dataset used during the training and help the model to generalize to avoid overfitting
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=12345)

# erase the memory from the unused variables 
del fashion_test_df,fashion_train_df,testing,training,W_grid,L_grid,n_training,index,i,axes
# Now form the data in order to feed the network 
# to do so, we need to reshape the data matrix into 28*28*1 which the shape accpeted by convolutional layer

# image format 
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

X_train.shape
#(48000, 28, 28, 1) it's 48000 samples, each one is 28*28*1 which grayscale 

# Now we are redy for training 
# import NN library 
import keras
# Now we will import all participate layers like (sequential convolutional layers, Maxs pooling layers, Dense, Flatten, Dropout .. )
from keras.models import Sequential # we build our model as a sequential form[convolutional lyer --> max pooling --> dropout --> flateten --> dense]
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model = Sequential()
cnn_model.add(Conv2D(64,3, 3, input_shape = (28,28,1), activation='relu'))# in Conv2D we need to specify how many kernels and feature detectors that we gonna apply
# for Conv2D we have 32 feature detectors and each one with 3*3 shape 
# inputshpe is the size of each image 

# Now we add Max pooling layer with pool size 2*2 
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

# flatten the features 
cnn_model.add(Flatten())

# to make the model fully connetced we will add our dense 
# define output dimentions and activation function 
# hidden layer
cnn_model.add(Dense(output_dim = 32,activation='relu'))

# output layer 
cnn_model.add(Dense(output_dim = 10,activation='sigmoid'))
# Now we build the network with differnt layers 

# Now we compile the network by defining loss, optimizer, metrics 
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
# we use categorical loss function because we are categorizing dataset out of 10 samples.

# Now we fit our model by defining the number of epochs 
epochs = 50# how many times we are presenting our trainung set and updating the weights 

cnn_model.fit(X_train,
            y_train,
            batch_size = 512,
            nb_epoch = epochs,
            verbose = 1,
            validation_data = (X_validate, y_validate))

# The accuracy of model is 95% even without using the dropout 

# Now we evakuate the trained model using the evaluate function 
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
# the accuracy is 91% 

# to evaluate the model in details, we let the model predict the test labels and compare them with the true ones
predicted_classes = cnn_model.predict_classes(X_test)

# To do so we will build grid for randomly 25 picked images 

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # flatten axes array

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')# to avoid printing the number of pixels and dim

plt.subplots_adjust(wspace=0.5)

# check the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)


# Now to understand the accurcy per class we will use classification_report 
from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_names))

# some classes have low accurcy like class 6 79% others have high like 99% 
# to improve the model we could increase the number of kernels or features detectors from 32 to 64 or even to 128
# OR using dropout which regularization to avoid overfitting 
