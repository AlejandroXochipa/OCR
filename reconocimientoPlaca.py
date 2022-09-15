import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import os
#from redneuronal import modelo1
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 



def get_hog():
    winSize = (20,20) #tamaÃ±o de las imagenes
    blockSize = (8,8)
    blockStride =  (4,4)
    cellSize = (8,8)
    nbins = 9
    DerivAperture = 1
    winSigma = -1.
    histogramNormType = 0 #tipo de histograma predefinido
    L2HysThreshold = 0.2 #Umbral por el que se multiplica
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, DerivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    
    return hog;

def escalar(img, m, n):
    if m > n:
        imgNueva = np.uint8(255*np.ones((m,round((m-n)/2),3)))
        escalada = np.concatenate((np.concatenate((imgNueva,img),axis=1),imgNueva),axis=1)
    else:
        imgNueva = np.uint8(255*np.ones((round((n-m)/2),n,3)))
        escalada = np.concatenate((np.concatenate((imgNueva,img),axis=0),imgNueva),axis=0)#corregir axis
        
    img = cv2.resize(escalada,(20,20))
    return img


def obtenerDatos():
    posiblesclases =  ['0','1','2','3','4','5','6','7','8','9','A','B','C','D',
                       'E','F','G','H','J','K','L','M','N','P','Q','R','S','T',
                       'U','V','W','X','Y','Z']
    posiblesclases_minusculas  = [ 'a','b','c','d','e','f',
                       'g','h','i','j','k','l','m','n','o','p',
                       'r','s','t','u','v','x','y','z']
                       
    datos = []
    clases = []
    datos2 = []
    clases2 = []
    for i in range(1,26):
        for j in posiblesclases:
            img = cv2.imread(j+'-'+str(i)+".jpg")
            if img is not None:
                m,n,_ = img.shape
                if m != 20 or n != 20:
                    img = escalar(img,m,n)
                hog = get_hog()
                #print(hog.compute(img))
                datos.append(np.array(hog.compute(img)))
                #datos.append(hog.compute(img))
                clases.append(np.where(np.array(posiblesclases)==j)[0][0])

                
    #print("C2",len(clases))           
    #Se agrego esta linea 
    ruta = r"C:\Users\xochi\OneDrive\Escritorio\UAEM\OCR2\Banco de datos\Recibo letras 2\Recibo letras 2"
    for i in range(1,26):
        for j in posiblesclases_minusculas:
            img = cv2.imread(os.path.join(ruta, j+'-'+str(i)+".jpg"))
            if img is not None:
                m,n,_ = img.shape
                if m != 20 or n != 20:
                    img = escalar(img,m,n)
                hog = get_hog()
                #print(hog.compute(img))
                datos.append(np.array(hog.compute(img)))
                #datos.append(hog.compute(img))
                clases.append(np.where(np.array(posiblesclases_minusculas)==j)[0][0])
    
                
                
                
                
                
    datos = np.array(datos) #[:,:,0]
    print("C1",len(clases))
    clases = np.array(clases)
    return datos, clases
                         




def modelo1():
    model = Sequential()
    model.add(Dense(200, input_dim=144))
    model.add(Dense(180))
    model.add(Dense(150))
    model.add(Dense(34, activation='softmax'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    return model




def clasificadorCaracteres():
    datos, clases = obtenerDatos()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(datos,clases)
    SVM = svm.SVC(kernel = 'linear', probability = True, 
                  random_state = 0, gamma = 'auto')
    SVM.fit(datos, clases)
    gnb = GaussianNB()
    gnb = gnb.fit(datos, clases)
    randomTree = RandomForestClassifier(max_depth=100, random_state=0)
    randomTree = randomTree.fit(datos,clases)
    #rn = modelo1()
    #rn.fit(datos, clases, epochs=100, batch_size = 100)
    return knn, SVM, gnb, randomTree                 




datos, clases = obtenerDatos()
# Evaluar el algoritmo KNN utilizando el metodo de validacion Hold Out
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(datos,clases,
                                        test_size=0.2, random_state=np.random)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)
rendimientoEntrenamientoKnn = (knn.score(X_train, Y_train)) * 100
print("Rendimiento de entrenamiento: " 
      + str(round(rendimientoEntrenamientoKnn,2))+"%")
rendimientoPruebaKnn = (knn.score(X_test, Y_test)) * 100
print("Rendimiento de entrenamiento: " 
      + str(round(rendimientoPruebaKnn,2))+"%")
 




