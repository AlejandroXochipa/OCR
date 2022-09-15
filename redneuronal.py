import cv2
import tensorflow as tf 
import numpy as np 
from reconocimientoPlaca import obtenerDatos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

def modelo1():
    model = Sequential()
    model.add(Dense(200, input_dim=144))
    model.add(Dense(180))
    model.add(Dense(150))
    model.add(Dense(34, activation='softmax'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    return model

def modelo2():
    model2 = Sequential()
    model2.add(Dense(200, input_dim=144))
    model2.add(Dense(180))
    model2.add(Dense(150))
    model2.add(Dense(34, activation='softmax'))
    model2.compile(loss = 'mse', optimizer = 'sgd', metrics = ['accuracy'])
    return model2

def modelo3():
    model3 = Sequential()
    model3.add(Dense(200, input_dim=144))
    model3.add(Dense(180))
    model3.add(Dense(150))
    model3.add(Dense(34, activation='linear'))
    model3.compile(loss = 'mean_squared_error', optimizer = 'Adadelta', metrics = ['accuracy'])
    return model3

def modelo4():
    model4 = Sequential()
    model4.add(Dense(200, input_dim=144))
    model4.add(Dense(180))
    model4.add(Dense(150))
    model4.add(Dense(34, activation='relu'))
    model4.compile(loss = 'mean_squared_logarithmic_error', optimizer = 'SGD', metrics = ['accuracy'])
    return model4

model = modelo1()
model2 = modelo2()
model3 = modelo3()
model4 = modelo4()
datos, clases = obtenerDatos()
numClases = 34
#Metodo de validacion HoldOut 
X_train, X_test, Y_train, Y_test= train_test_split(datos, clases, test_size=0.2, random_state=np.random)

Y_trainOneHot = tf.one_hot(Y_train, numClases)
Y_testOneHot = tf.one_hot(Y_test, numClases)
#Fase de aprendizaje
#model.fit(X_train, Y_trainOneHot, epochs=100, batch_size = 100)
model2.fit(X_train, Y_trainOneHot, epochs=50, batch_size = 50)
model3.fit(X_train, Y_trainOneHot, epochs=100, batch_size = 100)
model4.fit(X_train, Y_trainOneHot, epochs=100, batch_size = 100)
#Fase de prueba
#prediccion = model.predict(X_test)
prediccion2 = model2.predict(X_test)
prediccion3 = model3.predict(X_test)
prediccion4 = model4.predict(X_test)
#Y_pred = np.argmax(prediccion, 1)
Y_pred2 = np.argmax(prediccion2, 1)
Y_pred3 = np.argmax(prediccion3, 1)
Y_pred4 = np.argmax(prediccion4, 1)
#rendimientoPrueba = 100 + (1-np.sum(Y_pred==Y_test)/len(Y_test))
rendimientoPrueba2 = 100 + (1-np.sum(Y_pred2==Y_test)/len(Y_test))
rendimientoPrueba3 = 100 + (1-np.sum(Y_pred3==Y_test)/len(Y_test))
rendimientoPrueba4 = 100 + (1-np.sum(Y_pred4==Y_test)/len(Y_test))
#print("El rendimiento de la clasificacion es: " + str(round(rendimientoPrueba,1)))
print("El rendimiento de la clasificacion 2 es: " + str(round(rendimientoPrueba2,1)))
print("El rendimiento de la clasificacion 3 es: " + str(round(rendimientoPrueba3,1)))
print("El rendimiento de la clasificacion 4 es: " + str(round(rendimientoPrueba4,1)))

