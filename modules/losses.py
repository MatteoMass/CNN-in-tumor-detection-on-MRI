# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:13:48 2022


@author: Matteo Massetti
@Email: m.massetti2@studenti.unipi.it
"""


from metrics import tversky, dice_coef

def dice_loss(y_true, y_pred):
    '''
    compute the dice loss using dice_coeff method
    '''
    return 1-dice_coef(y_true, y_pred)


def tversky_Loss(y_true, y_pred, alpha = 1, beta = 3):
    '''
    compute the dice loss using tversky method
    '''
    return 1 - tversky(y_true, y_pred, alpha, beta)


