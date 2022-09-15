#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import ndimage
from reconocimientoPlaca import get_hog, escalar, clasificadorCaracteres
from deteccionPlaca import detectarPlaca
from Imagen import leerImagen


img = cv2.imread(r"C:\Users\xochi\OneDrive\Escritorio\UAEM\OCR2\Banco de datos\Prueba1.jpg")
#img = cv2.imread("car6.jpg")
#Esta línea regresa la placa segmentada
#placa = detectarPlaca(img)
placa = leerImagen(img)

placaGris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
umbral, _ = cv2.threshold(placaGris, 0, 255, cv2.THRESH_OTSU)

mascara = np.uint8(255*(placaGris<umbral))
output = cv2.connectedComponentsWithStats(mascara, 4, cv2.CV_32S)
cantidadObjetos = output[0]
etiquetas = output[1]
stats = output[2]

for i in range(1, cantidadObjetos):
    if stats[i,4] < stats[:,4].mean()/10:
        etiquetas = etiquetas - i*(etiquetas == i)

mascara = np.uint8(255*(etiquetas > 0))

#Dilatacion de caracteres de la placa
kernel = np.ones((3,3), np.uint8)
mascara = np.uint8(255 * ndimage.binary_fill_holes
                   (cv2.dilate(mascara,kernel)))

#Separar las letras de la placa con Bounding Rect
contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
caracteres = []
orden = []
placaCopia = placa.copy()

for cnt in contornos:
    x,y,w,h = cv2.boundingRect(cnt)
    caracteres.append(placa[y:y+h, x:x+w,:])
    orden.append(x)
    cv2.rectangle(placaCopia,(x,y),(x+w, y+h),(0,0,255),1)
    
caracteresOrdenados = [x for _,x in sorted(zip(orden,caracteres))]

#Fase de clasificacion
palabrasKnn = ""
palabrasSVM = ""
palabrasGnb = ""
palabrasrandomTree = ""
hog = get_hog()
knn, SVM, gnb, randomTree = clasificadorCaracteres()
posiblesClases = ['0','1','2','3','4','5','6','7','8','9',
              'A','B','C','D','E','F','G','H','J','K',
              'L','M','N','P','Q','R','S','T','U','V',
              'W','X','Y','Z']#,'a','b','c','d','e','f',
              #'g','h','i','j','k','l','m','n','p',
              #'r','s','t','u','v','x','y','z']
posiblesClases = np.array(posiblesClases)
caracteresPlaca = []

for i in caracteresOrdenados:
    m,n,_ = i.shape
    imagenEscalada = escalar(i,m,n)
    caracteresPlaca.append(imagenEscalada)
    caracteristicasImagen = np.array(hog.compute(imagenEscalada))
    palabrasKnn += posiblesClases[knn.predict([caracteristicasImagen])][0][0]
    palabrasSVM += posiblesClases[SVM.predict([caracteristicasImagen.T])][0][0]
    palabrasGnb += posiblesClases[gnb.predict([caracteristicasImagen.T])][0][0]
    palabrasrandomTree += posiblesClases[randomTree.predict([caracteristicasImagen.T])][0][0]
print("El clasificador knn da como resultado: " + palabrasKnn)
print("El clasificador SVM da como resultado: " + palabrasSVM)
print("El clasificador Gaussian da como resultado: " + palabrasGnb)
print("El clasificador Random Forest da como resultado: " + palabrasrandomTree)
cv2.putText(img, "La placa es: " + palabrasKnn, 
            (10,300), cv2.FONT_HERSHEY_DUPLEX,0.8,(0,255,255),1)
cv2.putText(img, "La placa es: " + palabrasSVM, 
            (10,200), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)
cv2.putText(img, "La placa es: " + palabrasGnb, 
            (10,100), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),1)
cv2.putText(img, "La placa es: " + palabrasrandomTree, 
            (10,50), cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),1)
cv2.imshow("carro", img)
cv2.waitKey(0)
cv2.destroyAllWindows()











