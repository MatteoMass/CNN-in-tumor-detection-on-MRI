# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:14:24 2022


@author: Matteo Massetti
@Email: m.massetti2@studenti.unipi.it
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, BatchNormalization

from metrics import dice_coef



def convolutionalBlock(in_, filters, dropout):
    '''
    create 2 convolutional layers on top of in_, possibly with a dropout layer between them 
    '''
    x = Conv2D(filters, (3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(in_)
    if dropout!= 0: x = Dropout(dropout)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x

def encoderBlock(in_, filters, useDropout, useBatchNormalization):
    '''
    create an encoder block by stacking a convolutionalBlock on top of in_, followed by a MaxPooling layer
    '''
    x = convolutionalBlock(in_, filters, useDropout)
    if(useBatchNormalization):
        x = BatchNormalization()(x)
    p = MaxPooling2D((2,2))(x)
    return x,p

def decoderBlock(in_, filters, useDropout, skipConnection):
    '''
    creates a decoder block by stacking a deconvolution layer on top of in_, adding the skip connection and a convolutional block
    '''
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(in_)
    x = concatenate([x, skipConnection])
    x = convolutionalBlock(x, filters, useDropout)
    return x

def getUnetModel(IMG_SIZE, Path, Bridge, dropout = 0.1, Loss = 'binary_crossentropy', useBatchNormalization = False):
    '''
    :param IMG_SIZE: Dimension of the image (assumed to be square)
    :type IMG_SIZE: int
    :param Path: List of number of filters to be used in each encoder and decoder block 
    :type Path: List [int]
    :param Bridge: number of filters to be used in the bridge (connection between encoder and decoder part)
    :type Bridge: int
    :param dropout: dropout parameter to be included in the network , defaults to 0.1
    :type useDropout: float in [0,1], optional
    :param Loss: loss function to be used, defaults to 'binary_crossentropy'
    :type Loss: either string (if one of the keras default loss) or a function, optional
    :param useBatchNormalization: flag that indicates if a BatchNormalization layer has to be added before each AveragePooling layer
    :type: boolean
    
    :return: Unet model already compiled
    :rtype: Keras model

    '''
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))
    
    #list of layers to be used in the skip connections
    SkipConnections = []
  
    p = inputs
  
    for i in Path:
        x,p = encoderBlock(p, i, dropout, useBatchNormalization)
        SkipConnections.append(x)
        
    p = convolutionalBlock(p, Bridge, dropout)
    
    Path.reverse()
    SkipConnections.reverse()
    
    for i in range(len(Path)):
        p = decoderBlock(p, Path[i], dropout, SkipConnections[i])

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(p)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=Loss, metrics=['accuracy', dice_coef])
    #model.summary()
    
    return model
