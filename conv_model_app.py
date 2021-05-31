import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, ReLU
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential

%matplotlib inline
np.random.seed(1)

__author__ = "Dennis Biber"


def happyModel():
    """
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
    ## Conv2D with 32 7x7 filters and stride of 1
    ## BatchNormalization for axis 3
    ## ReLU
    ## Max Pooling 2D with default parameters
    ## Flatten layer
    ## Dense layer with 1 unit for output & 'sigmoid' activation

    model = Sequential()
    X_input = Input((64,64,3))
    model.add(X_input)
    X = ZeroPadding2D(padding=3)(X_input)
    model.add(ZeroPadding2D(padding=3))
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    model.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0'))
    X = BatchNormalization(axis=3, name='bn0')(X)
    model.add(BatchNormalization(axis=3, name='bn0'))
    X = tf.nn.relu(X, name="ReLU")
    model.add(ReLU())
    X = MaxPooling2D(name='max_pool')(X)
    model.add(MaxPooling2D(name='max_pool'))

    X = Flatten()(X)
    model.add(Flatten())
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model.add(Dense(1, activation='sigmoid', name='fc'))
    
    return model


def convolutional_model(input_shape):
    """
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    model = Sequential()
    X = Conv2D(8, (4, 4), strides=(1, 1), padding="SAME", name='Z1')(input_img)
    model.add(Conv2D(8, (4, 4), strides=(1, 1), padding="SAME", name='Z1'))
    X = ReLU()(X)
    X = MaxPooling2D((8, 8), padding="SAME", strides=(8,8), name='P1')(X)
    X = Conv2D(16, (2, 2), padding="SAME", strides=(1, 1), name='Z2')(X)
    X = ReLU()(X)
    X = MaxPooling2D((4, 4), padding="SAME", strides=(4,4), name='P2')(X)

    X = Flatten()(X)
    X = Dense(6, activation='softmax', name='f')(X)

    model = Model(inputs=input_img, outputs=X, name='convolutional_model')
    
    return model


def main():

	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

	# Normalize image vectors
	X_train = X_train_orig/255.
	X_test = X_test_orig/255.

	# Reshape
	Y_train = Y_train_orig.T
	Y_test = Y_test_orig.T
	index = 124

	happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

	happy_model.summary()

	happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
	happy_model.evaluate(X_test, Y_test)
	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()

	index = 9
	plt.imshow(X_train_orig[index])

	X_train = X_train_orig/255.
	X_test = X_test_orig/255.
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T

	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
	test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
	history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)

	df_loss_acc = pd.DataFrame(history.history)
	df_loss= df_loss_acc[['loss','val_loss']]
	df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
	df_acc= df_loss_acc[['accuracy','val_accuracy']]
	df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
	df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
	df_acc.plot(title='	Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')


if __name__ == "__main__":
	main()