# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:43:52 2022


@author: Matteo Massetti
@Email: m.massetti2@studenti.unipi.it
"""
from PIL import Image
import os

SIZE = (64, 64)

filepath_x = "C:\\Users\\masse\\OneDrive\\Desktop\\DATASET\\immagini_x"
filepath_y = "C:\\Users\\masse\\OneDrive\\Desktop\\DATASET\\immagini_y"


destinationfolder_x = "C:\\Users\\masse\\OneDrive\\Desktop\\DATASET\\immagini_x_64\\"
destinationfolder_y = "C:\\Users\\masse\\OneDrive\\Desktop\\DATASET\\immagini_y_64\\"
for filename in os.listdir(filepath_x):
    image_x = Image.open(filepath_x + "\\" + filename)
    image_y = Image.open(filepath_y + "\\" + filename)
    
    
    image_x.thumbnail(SIZE)
    image_y.thumbnail(SIZE)
    
    image_x.save(destinationfolder_x+filename)
    image_y.save(destinationfolder_y+filename)
    
    
    