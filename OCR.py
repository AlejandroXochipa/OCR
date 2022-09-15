
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import ndimage
from reconocimientoPlaca import get_hog, escalar, clasificadorCaracteres
from deteccionPlaca import detectarPlaca
from Imagen import leerImagen
from imutils import contours












img = cv2.imread(r"C:\Users\xochi\OneDrive\Escritorio\UAEM\OCR2\Banco de datos\Prueba6.jpg")
#img = cv2.imread("car6.jpg")
#Esta línea regresa la placa segmentada
#placa = detectarPlaca(img)
placa = leerImagen(img)
#placa = detectarPlaca(placa)


def segmentarImagen(placa):
    placaGris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    umbral, _ = cv2.threshold(placaGris, 0, 255, cv2.THRESH_OTSU)
    mascara = np.uint8((placaGris<umbral)*255)
    num_labels, labels1,stats1,centroides = cv2.connectedComponentsWithStats(mascara, 4, cv2.CV_32S)
    valor_max_pixels = (np.max(stats1[:4][1:]))/2
    pin = np.where((stats1[:,4][1:]) > valor_max_pixels)
    pin = pin[0]+1
    mascaras = []
    mascara_final = 0
    #return stats1, pin
    #for i in range(0,len(pin)):
    #    mascara = pin[i] == labels1
    #    mascaras.append(mascara)
    #    mascara_final = mascara_final + mascaras[i]
    mascara = pin[-1] == labels1
    mascaras.append(mascara)
    mascara_final = mascara_final + mascaras[-1]
    mascarafinal2 = ndimage.binary_fill_holes(mascara_final).astype(int)
    mascarafinal2 = np.uint8(255 * mascarafinal2)
    maskObj = []
    maskConvex = []
    diferenciaArea = []
    
    contornos,_ = cv2.findContours(mascarafinal2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contornos[0]
    hull = cv2.convexHull(cnt)
    puntosConvex = hull[:,0,:]
    m,n = mascarafinal2.shape
    aux = np.zeros((m,n))
    mascaraConvex = np.uint8(255 * cv2.fillConvexPoly(aux,puntosConvex,1))
    maskConvex.append(mascaraConvex)
    #Comparar el area del ConvexHull vs Objeto
    areaObjeto = np.sum(mascara)/255
    areaConvex = np.sum(mascaraConvex)/255
    diferenciaArea.append(np.abs(areaObjeto - areaConvex))
    maskPlaca = maskConvex[np.argmin(diferenciaArea)]
    vertices = cv2.goodFeaturesToTrack(maskPlaca, 4, 0.01, 10)
    
    
    
    
    
    x = vertices[:,0,0]
    #x = vertices[0]
    y = vertices[:,0,1]
    vertices = vertices[:,0,:]
    xo = np.sort(x)
    yo = np.sort(y)

    xn = np.zeros((1,4))
    yn = np.zeros((1,4))
    n = (np.max(xo)-np.min(xo))
    m = (np.max(yo)-np.min(yo))

    xn = (x == xo[2]) * n + (x == xo[3]) * n
    yn = (y == yo[2]) * m + (y == yo[3]) * m
    verticesN = np.zeros((4,2))
    verticesN[:,0] = xn
    verticesN[:,1] = yn

    vertices = np.int64(vertices)
    verticesN = np.int64(verticesN)

    h, _ = cv2.findHomography(vertices, verticesN)
    placa = cv2.warpPerspective(placa,h, (np.max(verticesN[:,0]),
                                    (np.max(verticesN[:,1]))))
    
    
    
    

    #vertices = cv2.goodFeaturesToTrack(mascara_final, 4, 0.01, 10)
    return vertices,mascarafinal2, placa, mascaraConvex




#Nueva funcion
def segmentacion2(img): 
    image = img #cv2.imread('1.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([37, 2, 0], np.uint8)
    upper = np.array([179, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    #num_labels, labels1,stats1,centroides = cv2.connectedComponentsWithStats(mask, cv2.CV_32S)
    #stats_complex = stats1[2:][4:]
    # Create horizontal kernel and dilate to connect text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    #kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(mask, kernel, iterations=5)

    # Find contours and filter using aspect ratio
    # Remove non-text contours by filling in the contour
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 5:
            cv2.drawContours(dilate, [c], -1, (0,255,0), -1)

    # Bitwise dilated image with mask, invert, then OCR
    result = 255 - cv2.bitwise_and(dilate, mask)
    #data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
    #print(data)
    #mascarafinal = 0
    #mascarafinal = ndimage.binary_fill_holes(result).astype(int)
    #mascarafinal = np.uint8(1 * mascarafinal)
    cv2.imshow('Mascara', mask)
    #cv2.imshow('dilate', dilate)
    #cv2.imshow('result', result)
    #cv2.imshow('mascara final', mascarafinal)
    #cv2.waitKey()
    return result



def nothing(x):
    pass



def hsv(image):
    cv2.imshow('umbral', image)
    cv2.namedWindow('image')
    cv2.createTrackbar('HMin', 'image', 0, 180, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 180, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 180)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        
        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

    # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    
import pytesseract

    

def quitarfondo(img):
    image = img #cv2.imread('image.png')
    # create grayscale
    tam_contorno = []
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perform threshold
    retr , thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 255])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # find contours
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create emtpy mask
    #mask = np.zeros(image.shape[:2], dtype=image.dtype)

    # draw all contours larger than 20 on the mask
    for c in contours:
        if cv2.contourArea(c) < 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(mask, [c], 0, (255), -1)

# apply the mask to the original image
    result = 255 - cv2.bitwise_and(image,image, mask= mask)
    #cv2.imshow("Resultadossosiso", result)
    #cv2.imshow("RRRR ", thresh)
    return result
        #if cv2.contourArea(c) > 20:
            #x, y, w, h = cv2.boundingRect(c)
            #cv2.drawContours(mask, [c], 0, (255), -1)
    
    """img = img #cv2.imread("py.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)

    for i in img_contours:

        if cv2.contourArea(i) > 100:

            break
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [i],-1, 255, -1)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Original Image", img)
    cv2.imshow("Image with background removed", new_img)"""
    #return tam_contorno

#Comentada

def segmentacion3(img):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            x,y,w,h = cv2.boundingRect(c)
            ROI = 255 - image[y:y+h, x:x+w]
            cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
            ROI_number += 1
            #cv2.imshow('threshxedddd', thresh)
            #cv2.imshow('imagexdddd', image)
            #cv2.waitKey()




vertices, p1, plaquisima, mascaraConvex = segmentarImagen(placa)
placa = plaquisima
cv2.imshow("Segmentacion I", plaquisima)
placa_copia = placa.copy()

segmentacion3(placa)
#rectangulo = np.max(np.array(quitarfondo(plaquisima)))
#Easyocr
#reader = easyocr.Reader(["es"])
#resultado = reader.readtext(placa_copia)

#texto = clasificador(placa)



placaGris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
umbral, _ = cv2.threshold(placaGris, 0, 255, cv2.THRESH_OTSU)



#hdeetor = cv2.cvtColor(umbralizacion_1, cv2.COLOR_BGR2HSV)
rectangulo = quitarfondo(placa)
seg2 = segmentacion2(placa) #Aqui esta la segmentacion 
#hsv(placa)



_ , umbralizacion_1 = cv2.threshold(placaGris, 127, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)


num_labelsh, labelsh,statsh,centroidesh = cv2.connectedComponentsWithStats(umbralizacion_1, 4, cv2.CV_32S)
kernelh = np.ones((5,5),np.uint8)
dilateh = cv2.dilate(umbralizacion_1, kernelh, iterations=1)
outputh = cv2.connectedComponentsWithStats(dilateh, 4, cv2.CV_32S)
cantidadObjetosh = outputh[0]
etiquetash = outputh[1]
statsh = outputh[2]
cv2.imshow("Dilatacion", dilateh)




pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#result = 255 - cv2.bitwise_and(dilate, mask)
datax = pytesseract.image_to_string(placa)



#Separar las letras de la placa con Bounding Rect
contornosh, _ = cv2.findContours(dilateh, cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
caracteresh = []
ordenh = []
placaCopiah = placa.copy()

for cnt in contornosh:
    x,y,w,h = cv2.boundingRect(cnt)
    caracteresh.append(placa[y:y+h, x:x+w,:])
    ordenh.append(x)
    cv2.rectangle(placaCopiah,(x,y),(x+w, y+h),(0,0,255),1)
    #cv2.rectangle(placaCopiah, (x,y), (x+w,y+h), (0,0,255),2)
    #cv2.imshow("Rectangulos XDDDD", placaCopiah)
#caracteresOrdenadosh = [x for _,x in sorted(zip(ordenh,caracteresh))]

























mascara = np.uint8(255*(placaGris<umbral))
output = cv2.connectedComponentsWithStats(mascara, 4, cv2.CV_32S)
cantidadObjetos = output[0]
etiquetas = output[1]
stats = output[2]
stats_complex = stats[2:][4:]


"""
ret, thresh = cv2.threshold(placaGris,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Eliminación del ruido
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
 
# Encuentra el área del fondo
sure_bg = cv2.dilate(opening,kernel,iterations=3)
 
# Encuentra el área del primer
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
 
# Encuentra la región desconocida (bordes)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Etiquetado
ret, markers = cv2.connectedComponents(sure_fg)
 
# Adiciona 1 a todas las etiquetas para asegura que el fondo sea 1 en lugar de cero
markers = markers+1
 
# Ahora se marca la región desconocida con ceros
markers[unknown==255] = 0
markers = cv2.watershed(placa_copia,markers)
placa_copia[markers == -1] = [255,0,0]

#mascara_chida= ndimage.binary_fill_holes(markers).astype(int)
#mascara_chida = np.uint8(255 * mascara_chida)    
    
    
#contornos2, _ = cv2.findContours(mascara, cv2.RETR_TREE, 
                                  # cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(placa, contornos2, -1, (0,255,0), 3)
cv2.imshow("Contornos", placa_copia)
"""





maskConvex1 = []
diferenciaArea1 = []
ret, thresh = cv2.threshold(placaGris,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contorno_p ,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contorno_p[0]
hull1 = cv2.convexHull(cnt)
puntosConvex1 = hull1[:,0,:]
m,n = thresh.shape
aux = np.zeros((m,n))
mascaraConvex1 = np.uint8(255 * cv2.fillConvexPoly(aux,puntosConvex1,1))
maskConvex1.append(mascaraConvex1)



#gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
ret, binarizada = cv2.threshold(placaGris, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
caracteres3 = []
orden2 = []

contorno_c , jerarquia_c = cv2.findContours(binarizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for casa in contorno_c:
    x,y,w,h = cv2.boundingRect(casa) #Calcula el punto de origen
    cv2.rectangle(placa_copia, (x,y), (x+w,y+h), (0,0,255),2)
    cv2.imshow("Rectangulos aproximados", placa_copia)

for c in contorno_c:
    #Calcular la precision del contorno calculado
    precision = 0.03 * cv2.arcLength(c, True) #Calcula el perimetro de la curva
    approx = cv2.approxPolyDP(c, precision, True) #Dibuja un poligono derecho
    x,y,w,h = cv2.boundingRect(approx)
    caracteres3.append(placa[y:y+h, x:x+w,:])
    orden2.append(x)
    cv2.drawContours(placa_copia, [approx], 0, (0,255,0), 3)
    cv2.rectangle(placa_copia,(x,y),(x+w, y+h),(255,0,0),1)
    cv2.imshow("Approx Poly", placa_copia)
#Quitar los dos primeros stats y rellenar los agujeros que queden de los demas objetos 

#valor_mayor = np.max(stats_complex[:][5:])
#stats_c = [c for c in stats_complex if c[]]

#for s in stats_complex:
    
print("Soy el stat", len(stats_complex))
#prueba = []
#prueba2 = []
for i in range(0, len(stats_complex)):
    #print(stats_complex[i,4])
    #prueba.append(stats_complex[i,4])
    #prueba2.append(stats_complex[:,4].mean())
    if stats_complex[i,4] < stats_complex[:,4].mean()/len(stats_complex):
        etiquetas = etiquetas - i*(etiquetas == i)
#len(stats_complex)
mascara = np.uint8(255*(etiquetas > 0))


contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
caracteres = []
orden = []
placaCopia = placa.copy()

for cnt in contornos:
    #x,y,w,h = cv2.boundingRect(cnt)
    #caracteres.append(placa[y:y+h, x:x+w,:])
    #orden.append(x)
    #cv2.rectangle(placaCopia,(x,y),(x+w, y+h),(0,0,255),1)
    #cv2.drawContours(placa_copia, [approx], 0, (0,255,0), 3)
    #cv2.imshow("Caracteres", placaCopia)
    x,y,w,h = cv2.boundingRect(cnt)
    caracteres.append(placa[y:y+h, x:x+w,:])
    orden.append(x)
    cv2.rectangle(placaCopia,(x,y),(x+w, y+h),(0,0,255),1)
    cv2.imshow("Caracteres", placaCopia)
    
palabrasSvm = datax
"""diccionario = {}
diccionario2 = {}
contador = 0
contadorx = 0
for i in orden:
    diccionario[contador] = i
    contador+= 1
    
for h in sorted(orden):
    diccionario2[contadorx] = h
    contadorx+=1
    
    
    
    
pruebita = sorted(list(np.array(zip(orden,caracteres)).all()))   
#x9, y9 = zip(sorted(orden), sorted(caracteres))
#orde = set(orden)
ojala = []
ct = np.array(caracteres)
#pc = ct.argsort()
#xd = (sorted(np.all(zip(orden,caracteres))))
dar = np.array((list(zip(np.array(orden),np.array(caracteres)))))
tu = np.sort((dar[:,0]))
sepa = np.sort(list(zip(np.array(orden),np.array((caracteres)))), axis=0)
print("Soy la lista", np.sort(list(zip(np.array(orden),np.arange(126))), axis=0))
#caracteresOrdenados = [x for _,x in np.sort(list(zip(orden,caracteres)), axis=0)]
caracteresOrdenados = [x for _,x in np.sort(list(zip(np.array(orden),np.arange(126))), axis=0)]
#contador = 0
#for i,x in ((zip(orden,np.array(caracteres)))):
#    ojala.append([i,x])
#ojala = np.array(ojala).reshape(126,2)
#ojala2 = np.all(sorted(zip(ojala[:,0],ojala[:,1])))
#ojala = sorted(ojala)
#ojala = sorted(np.array(zip(ojala)).all())
#ojala0 = sorted(ojala[]).any()

x9 = sorted(orden)
w9 = sorted(np.array(caracteres).reshape((126,1))[2:])
print("Soy las dimensiones", np.array(caracteres).shape)
y9 = sorted(caracteres)
caracteresOrdenados = [x for _,x in sorted(zip(np.array(orden).any(axis=0),np.array(caracteres).all(axis=0)))]
"""
caracteresOrdenados = caracteres
#caracteresOrdenadosChidos = [x for _,x in sorted(zip(np.array(orden2).all(),np.array(caracteres3).all()))]






"""
Stats para contornos y rellenar los huecos de los stats encontrados del proxxy
"""






"""
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




#cnt = contornos[0]
#hull = cv2.convexHull(cnt)




caracteres = []
orden = []
placaCopia = placa.copy()

for cnt in contornos:
    x,y,w,h = cv2.boundingRect(cnt)
    caracteres.append(placa[y:y+h, x:x+w,:])
    orden.append(x)
    cv2.rectangle(placaCopia,(x,y),(x+w, y+h),(0,0,255),1)
    
pruebita = sorted(zip(orden,caracteres))
caracteresOrdenados = [x for _,x in sorted(zip(orden,caracteres))]


#contador = 0
#for i, k in zip(orden, caracteres):
    
#    contador += 1
listaaaaa = []
for x,y in sorted(zip(orden,caracteres)):
    listaaaaa.append(y)

#ccc = [x for _,x in (zip(sorted(orden),sorted(caracteres)))]
#y5 = sorted(orden)"""
#Fase de clasificacion
palabrasKnn = ""
palabrasSVM = ""
palabrasGnb = ""
palabrasrandomTree = ""
palabrasRedNeuronal = ""
hog = get_hog()
knn, SVM, gnb, randomTree = clasificadorCaracteres()
posiblesClases = ['0','1','2','3','4','5','6','7','8','9',
              'A','B','C','D','E','F','G','H','J','K',
              'L','M','N','P','Q','R','S','T','U','V',
              'W','X','Y','Z', 'a','b','c','d','e','f',
              'g','h','i','j','k','l','m','n','o','p',
              'r','s','t','u','v','x','y','z']


"""posiblesClases  = [ 'a','b','c','d','e','f',
                   'g','h','i','j','k','l','m','n','o','p',
                   'r','s','t','u','v','x','y','z']"""


posiblesClases = np.array(posiblesClases)
caracteresPlaca = []
#Prediction_CWA = np.expand_dims(posiblesClases, axis=0)
history = ""
for i in caracteresOrdenados:
    m,n,_ = i.shape
    imagenEscalada = escalar(i,m,n)
    caracteresPlaca.append(imagenEscalada)
    caracteristicasImagen = np.array(hog.compute(imagenEscalada))

    palabrasSVM += posiblesClases[SVM.predict([caracteristicasImagen.T])][0][0]

#print("El clasificador knn da como resultado: " + palabrasKnn)
print("El clasificador SVM da como resultado: " + palabrasSvm, end="")
#print("El clasificador Gaussian da como resultado: " + palabrasGnb)
#print("El clasificador Random Forest da como resultado: " + palabrasrandomTree)
#print("El clasificador Red Neuronal da como resultado: " + palabrasRedNeuronal)
#cv2.putText(img, "La placa es: " + palabrasKnn, 
#            (10,300), cv2.FONT_HERSHEY_DUPLEX,0.8,(0,255,255),1)
cv2.putText(img, "La placa es: " + palabrasSvm, 
            (10,200), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)
#cv2.putText(img, "La placa es: " + palabrasGnb, 
#            (10,100), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,0),1)
#cv2.putText(img, "La placa es: " + palabrasrandomTree, 
 #           (10,50), cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),1)
cv2.imshow("Recibo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




f = open ('TextoRecibo.txt','w')
f.write(palabrasSvm)
f.close()






