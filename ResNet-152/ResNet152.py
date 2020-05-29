"""Implementation of ResNet-152, described in K. He, X. Zhang, S. Ren, J. Sun 
"Deep Residual Learning for Image Recognition" (arXiv:1512.03385v1[cs.CV])
batch size is 4, because net was trained on a cheap laptop
Dataset obtained from deeplearning.io's "Deep Learning" specialization (Course 4)
"""


import numpy as np
import h5py
import os
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.clear_session()
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


CONFIG = {
          "input_size":{
            "width": 64,
            "height": 64, 
            "channels": 3
          },
          "batch_size":{
            "train": 2,
            "val": 2   
          },
        "epochs": 100,
        "lr": 0.0001
}

def load_dataset():
    """A function for loading dataset and preprocessing"""

    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def starting_block(X, f, filters, strides, maxpool_size, stage=1):
    """Imlementation of starting block of ResNet
    X - input tensor (m, input_size["width"], input_size["height"], n_C)
    f - integer, specifying the shape of the CONV's window
    filters - integer, specifying numbers of filters in CONV layer
    strides - integer, specifying strides of layers
    maxpool_size - integer, specifying the shape of MAXPOOl' window
    stage - integer, used to name layers
    block - a string character, used to name positions

    X - output of identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Defining name bias
    conv_name_base = 'conv' + str(stage) + '_branch'
    pool_name_base = 'pool' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    # First convolutional layer 
    X = Conv2D(filters=filters, kernel_size=(f, f), strides=strides, name=conv_name_base, kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=maxpool_size, strides=strides, padding='valid', name=pool_name_base)(X)
    
    return X


def identity_block(X, f, filters, stage, block):
    """Implementation of identity block
    X - input tensor, shape of (m, n_H_prev, n_W_prev, n_C_prev)
    f - integer, specifying the shape of the middle CONV's window for the main path
    filters - list of integers, specifying shapes of filters for the main path
    stage - integer, used to name layers
    block - a string character, used to name position 

    X - output of identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Defining name bias
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Get filters
    F1, F2, F3 = filters

    # Save input value for residue block
    X_residue = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '_2a', kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '_2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '_2b', kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '_2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '_2c', kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '_2c')(X)
    
    # Adding residue
    X = Add()([X, X_residue])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    """Implementation of a convolutional block
    X - input tensor shape of (m, n_H_prev, n_W_prev, n_C_prev)
    f - integer, specifying the shape of middle CONV's window for the main path
    filters - list of integers, defining the number of filters in the CONV layers of the main path
    stage - integer, used to name layers, depending on the position in the network
    block - string/character, used to name the layers, depending on the position in the network
    s - integer, specifying the stride to be used

    X - output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    # defining name bias
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # retrieve filters' size
    F1, F2, F3 = filters

    # Save input value
    X_residue = X

    # first component of the path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '_2a', kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # second component of the path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '_2b', kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # third component of the path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '_2c', kernel_initializer=glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Residue path
    X_residue = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '_1', kernel_initializer=glorot_uniform(seed=1))(X_residue)
    X = BatchNormalization(axis=3, name=bn_name_base + '1')(X_residue)

    # Adding residue to the main path
    X = Add()([X, X_residue])
    X = Activation('relu')(X)

    return X

def ResNet152 (input_shape=(64, 64, 3), classes=6):
    """Implementation of ResNet152 following cited paper's acrhitecture. First block of layer substituted with convolutional block.

    input_shape - shape of the images of the dataset
    classes - integer, number of classes

    model - Keras Model() instance
    """

    # Define input as a tensor of shape input_shape
    X_input = Input(input_shape)

    # Zero-padding of input
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = starting_block(X, 7, 64, 2, 3)

    # Stage 2
    num_filters = [64, 64, 256]
    X = convolutional_block(X, 3, num_filters, stage=2, block='a', s=1)
    X = identity_block(X, 3, num_filters, stage=2, block='b')
    X = identity_block(X, 3, num_filters, stage=2, block='c')

    # Stage 3
    num_filters = [128, 128, 512]
    X = convolutional_block(X, 3, num_filters, stage=3, block='a')
    X = identity_block(X, 3, num_filters, stage=3, block='b')
    X = identity_block(X, 3, num_filters, stage=3, block='c')
    X = identity_block(X, 3, num_filters, stage=3, block='d')
    X = identity_block(X, 3, num_filters, stage=3, block='e')
    X = identity_block(X, 3, num_filters, stage=3, block='f')
    X = identity_block(X, 3, num_filters, stage=3, block='g')
    X = identity_block(X, 3, num_filters, stage=3, block='h')

    # Stage 4
    num_filters = [256, 256, 1024]
    X = convolutional_block(X, 3, num_filters, stage=4, block='a')
    X = identity_block(X, 3, num_filters, stage=4, block='b')
    X = identity_block(X, 3, num_filters, stage=4, block='c')
    X = identity_block(X, 3, num_filters, stage=4, block='d')
    X = identity_block(X, 3, num_filters, stage=4, block='e')
    X = identity_block(X, 3, num_filters, stage=4, block='f')
    X = identity_block(X, 3, num_filters, stage=4, block='g')
    X = identity_block(X, 3, num_filters, stage=4, block='h')
    X = identity_block(X, 3, num_filters, stage=4, block='i')
    X = identity_block(X, 3, num_filters, stage=4, block='j')
    X = identity_block(X, 3, num_filters, stage=4, block='k')
    X = identity_block(X, 3, num_filters, stage=4, block='l')
    X = identity_block(X, 3, num_filters, stage=4, block='m')
    X = identity_block(X, 3, num_filters, stage=4, block='n')
    X = identity_block(X, 3, num_filters, stage=4, block='o')
    X = identity_block(X, 3, num_filters, stage=4, block='p')
    X = identity_block(X, 3, num_filters, stage=4, block='q')
    X = identity_block(X, 3, num_filters, stage=4, block='r')
    X = identity_block(X, 3, num_filters, stage=4, block='s')
    X = identity_block(X, 3, num_filters, stage=4, block='t')
    X = identity_block(X, 3, num_filters, stage=4, block='u')
    X = identity_block(X, 3, num_filters, stage=4, block='v')
    X = identity_block(X, 3, num_filters, stage=4, block='w')
    X = identity_block(X, 3, num_filters, stage=4, block='x')
    X = identity_block(X, 3, num_filters, stage=4, block='y')
    X = identity_block(X, 3, num_filters, stage=4, block='z')
    X = identity_block(X, 3, num_filters, stage=4, block='aa')
    X = identity_block(X, 3, num_filters, stage=4, block='ab')
    X = identity_block(X, 3, num_filters, stage=4, block='ac')
    X = identity_block(X, 3, num_filters, stage=4, block='ad')
    X = identity_block(X, 3, num_filters, stage=4, block='ae')
    X = identity_block(X, 3, num_filters, stage=4, block='af')
    X = identity_block(X, 3, num_filters, stage=4, block='ag')
    X = identity_block(X, 3, num_filters, stage=4, block='ah')
    X = identity_block(X, 3, num_filters, stage=4, block='ai')
    X = identity_block(X, 3, num_filters, stage=4, block='aj')

    # Stage 5
    num_filters = [512, 512, 2048]
    X = convolutional_block(X, 3, num_filters, stage=5, block='a')
    X = identity_block(X, 3, num_filters, stage=5, block='b')
    X = identity_block(X, 3, num_filters, stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=1))(X)

    # Create Model
    model = Model(input=X_input, outputs=X, name='ResNet152')

    return model

# Compile model
model = ResNet152(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load datasets
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# Fit model on a dataset
model.fit(X_train, Y_train, epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"]["train"])

# Make predictions
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Accuracy = " + str(preds[1]))

# Save model
model.summary()
plot_model(model, to_file='model.png')
model.save('ResNet152.h5')
SVG(model_to_dot(model).create(prog='dot', format='svg'))










