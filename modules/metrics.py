# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:12:33 2022


@author: Matteo Massetti
@Email: m.massetti2@studenti.unipi.it
"""

import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    '''
    return the dice coefficient between y_true and y_pred, the smoothing is used to avoid division by zero
    '''
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def tversky (y_true, y_pred, alpha = 1, beta = 3, smooth=1e-6):
    '''
    return the tversky index between y_true and y_pred, the smoothing is used to avoid division by zero
    alpha weights False Positive while beta False Negative
    '''
    inputs = K.flatten(y_true)
    targets = K.flatten(y_pred)
    
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
       
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth) 
    return Tversky