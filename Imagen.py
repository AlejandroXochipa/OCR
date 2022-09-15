# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:24:31 2022

@author: xochi
"""

import cv2

def leerImagen(imagen):
    print(imagen.shape)
    #recortada = imagen[127:250,26:222]
    #recortada = imagen[129:153,27:113]
    #imagen = cv2.rectangle(imagen,(167,225),(273,247),(0,0,0),3)
    #imagen = cv2.rectangle(imagen,(21,132),(238,250),(0,0,0),1)
    #recortada = imagen[228:260,169:280]
    imagen = cv2.rectangle(imagen,(25,128),(336,264),(0,0,0),1)
    
    recortada = imagen[13:288,8:638]
    #recortada = imagen[134:279,62:376]
    cv2.imshow("Recortada", recortada)
    return recortada