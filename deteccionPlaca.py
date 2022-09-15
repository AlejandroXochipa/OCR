import cv2
import numpy as np
from scipy import ndimage

def detectarPlaca(img):
    #img = cv2.rectangle(img,(129,160),(27,113),(0,0,0),3)
    #cv2.imshow("xd", img)
    imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u,_ = cv2.threshold(imgGris, 0, 255, cv2.THRESH_OTSU)

    mascara = np.uint8(255*(imgGris>u))

    salida = cv2.connectedComponentsWithStats(mascara, 4, cv2.CV_32S)
    cantidadObjetos = salida[0]
    labels = salida[1]
    stats = salida[2]
    maskObj = []
    maskConvex = []
    diferenciaArea = []

    for i in range(1, cantidadObjetos):
        if stats[i,4] > stats[:,4].mean():
            mascara = ndimage.binary_fill_holes(labels==i) 
            mascara = np.uint8(255 * mascara)
            maskObj.append(mascara)
            #Calcular el convexHull de la placa del automovil
            contornos,_ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contornos[0]
            hull = cv2.convexHull(cnt)
            puntosConvex = hull[:,0,:]
            m,n = mascara.shape
            aux = np.zeros((m,n))
            mascaraConvex = np.uint8(255 * cv2.fillConvexPoly(aux,puntosConvex,1))
            maskConvex.append(mascaraConvex)
            #Comparar el area del ConvexHull vs Objeto
            areaObjeto = np.sum(mascara)/255
            areaConvex = np.sum(mascaraConvex)/255
            diferenciaArea.append(np.abs(areaObjeto - areaConvex))

    maskPlaca = maskConvex[np.argmin(diferenciaArea)]
    #print(maskPlaca)
    #Corrección de perspectiva
    vertices = cv2.goodFeaturesToTrack(maskPlaca, 4, 0.01, 10)
    print(vertices, "DIM")
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
    placa = cv2.warpPerspective(img,h, (np.max(verticesN[:,0]),
                                    (np.max(verticesN[:,1]))))
    cv2.imshow("Placota", placa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return placa




#cv2.imshow("Placa", placa)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


