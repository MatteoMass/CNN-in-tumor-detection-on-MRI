# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:04:55 2022


@author: Matteo Massetti
@Email: m.massetti2@studenti.unipi.it
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dropout, AveragePooling2D, Input, Add, UpSampling2D, Activation

from metrics import dice_coef

def convolutionalBlock(in_, filters, dropout):
    '''
    create 2 convolutional layers on top of in_, possibly with a dropout layer between them 
    '''
    x = Conv2D(filters, (3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(in_)
    if dropout!=0: x = Dropout(dropout)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x

def encoderBlock(in_, filters, dropout):
    '''
    create an encoder block by stacking a convolutionalBlock on top of in_, followed by a AveragePooling layer
    '''
    x = convolutionalBlock(in_, filters, dropout)
    p = AveragePooling2D((2,2))(x)
    return p


def getFCN(IMG_SIZE, Path, addMask, Bridge, dropout = 0.1, Loss = 'binary_crossentropy'):
    '''
    
    :param IMG_SIZE: Dimension of the image (assumed to be square)
    :type IMG_SIZE: int
    :param Path: List of number of filters to be used in each encoder and decoder block 
    :type Path: List [int]
    :param addMask: binary vector of the same size as Path indicating which layers are to be part of the skip connection
    :type addMask: List [0 or 1]
    :param Bridge: number of filters to be used in the bridge (connection between encoder and decoder part)
    :type Bridge: int
    :param dropout: dropout parameter to be included in the network , defaults to 0.1, defaults to 0.1
    :type dropout: float in [0,1], optional
    :param Loss: loss function to be used, defaults to 'binary_crossentropy'
    :type Loss: either string (if one of the keras default loss) or a function, optional
    
    :return: FCN model already compiled
    :rtype: Keras model

    '''
    
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))
    #list of layers to be used in the skip connections
    SkipConnections = []

    p = inputs

    for i in Path:
        p = encoderBlock(p, i, dropout)
        SkipConnections.append(p)
  
    p = convolutionalBlock(p, Bridge, dropout)
    p = Conv2D(1, (1, 1), activation='sigmoid')(p)
  
    SkipConnections.reverse()
  
    for i in range(len(addMask)):
        if addMask[i]:
            t = Conv2D(1, (1, 1), activation='sigmoid')(SkipConnections[i])
            p = Add()([p, t])
        p = UpSampling2D((2,2))(p)


    outputs = Activation(tf.nn.sigmoid)(p)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=Loss, metrics=['accuracy', dice_coef])
    #model.summary()
    
    return model