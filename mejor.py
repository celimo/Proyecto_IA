# Se importan librerias utilizadas en el código
import tensorflow as tf          # Se importa tensorflow
from tensorflow import keras     # Realizar el modelo de la NN
import pandas as pd              # Abrir los datos
import matplotlib.pyplot as plt  # Graficar curvas de validación
import numpy as np               # Para determinar las clases
from mlxtend.plotting import plot_confusion_matrix  # Para matriz confusion
from sklearn.metrics import confusion_matrix        # Para matriz confusion

# Manejo de datos

data = np.loadtxt('DatosSimplificados.txt') # Se cargan los datos obtenidos de la tarea 6

np.random.seed() # Crea una semilla para el random
data = np.random.permutation(data)

# Divide los datos entre train y validación
train, test = np.split(data, [int(0.8*len(data))])

# Se separan los datos para el entrenamiento
train_data = train[:, :7]
train_labels = train[:, 7]

# Se separan los datos para la validación
test_data = test[:, :7]
test_labels = test[:, 7]

# Normalización de datos

for i in range(len(train_data[0])):
  train_data[:, i] /= np.max(train_data[:, i])
  test_data[:, i] /= np.max(test_data[:, i])

# ================ clasificaciones  ================

class_names = ['Sano', '1-Enf', '2-Enf', '3-Enf', '4-Enf']

# ================ Se crea el modelo ================
# Se trabaja con 3 capas: 1 de entrada, 1 oculta y 1 de salida
# 13 entradas
# 4 percetrones en la capa oculta (este dato puede variar)
# 5 clasificaciones en la salida

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,)),       # input layers
    keras.layers.Dense(12, activation='sigmoid'),  # hidden layers
    keras.layers.Dense(2, activation='softmax')   # output layers
    ])

# ================ Se compila el modelo ================
# Se trabaja con el optimizador "adaptive moment estimation (ADAM)" y con
# el optimizador "Root Mean Square Prop (RMSprop)"
# La función de pérdida se configura para trabajar en clasificación
# Se va a medir la precisión de los datos
# El objeto optimizador indica el tipo con su tasa de aprendizaje y
# el momento a utilizar, en el caso del Adam, el momento es el beta_1

optimizador = keras.optimizers.Adam(learning_rate=0.001,
                                    beta_1=0.2)

# Se compila el modelo
# optimizer: Optimizador a usar
# loss: tipo de función de pérdida
# metrics: la métrica a evaluar durante el entrenamiento y el testing

model.compile(optimizer=optimizador,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ================ Se entrena el modelo con los datos ================

iteration = 3000  # Cantidad de iteraciones para el entrenamiento

# Se inicia  el entrenamiento de la red
# Los primeros parámetros son los datos de entrenamiento y las clasificaciones
# epochs: iteraciones del entrenamiento
# validation_split: Utiliza un 30% de los datos para validación
# validation_freq: cada cierta freqVal de iteraciones se hace la validación
# verbose: (0) No se muestran las iteraciones (1) muestra las iteraciones

print("Realizando entrenamiento...")

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      mode='min',
                                      verbose=1,
                                      patience=30,
                                      restore_best_weights=True)

training = model.fit(train_data, train_labels,
                     epochs=iteration,
                     validation_split=0.3,
                     validation_freq=1,
                     verbose=0,
                     callbacks=[es])

print("Fin entrenamiento")

# ================ Configuración para realizar la gráfica ================

loss = training.history['loss']  # Datos de la perida de entrenamiento
val_loss = training.history['val_loss']  # Datos de la perdida de validación
x = []  # Datos eje x para la curva de etrenamiento

for i in range(len(loss)):  # Se agregan datos al eje x de entrenamiento
    x.append(i + 1)

print(loss[-1], val_loss[-1])

plt.xlabel("Iteración")
plt.ylabel("Error")
plt.plot(x, loss, label="Pérdida de entrenamiento")
plt.plot(x, val_loss, label="Pérdida de validación")
plt.legend()
plt.savefig("Graph/CurvasEntrenamiento.png")

# ================ Guardar curvas de périda ================

file = open("curvas/LossTrain.txt", "w")
for i in range(len(loss)):
    file.write(str(x[i]) + " " + str(loss[i]) + "\n")
file.close()

file = open("curvas/ValidTrain.txt", "w")
for i in range(len(val_loss)):
    file.write(str(x[i]) + " " + str(val_loss[i]) + "\n")
file.close()

# ================ Crear matriz de confusión ================
# prob_matrix: Se calcula la probabilidad que el dato de entrada corresponda
#               a cierta clasificación
# pred_labels: Se obtiene el indice donde se encuentra la mayor probabilidad
#              este array es el que es comparado con el test_labels para
#              la matriz de confusión

prob_matrix = model.predict(test_data)
pred_labels = np.argmax(prob_matrix, axis=-1)

mat = confusion_matrix(test_labels, pred_labels)
plot_confusion_matrix(conf_mat=mat,
                      figsize=(8, 8),
                      show_normed=True,
                      cmap=plt.cm.Blues)

plt.xlabel("Valor predicho")
plt.ylabel("Valor real")
plt.savefig("MatrizConf.png")

# ================ Guardar los pesos del entrenamiento ================
# Se guardan los pesos para comparar si mejoraron el resultado

model.save_weights('model/weights.h5',
                   overwrite=True)

# ================ Guardar el modelo utilizado ================
# El modelo se guarda para comparar los resultados y en el cas que Sea
# mejor al anterior se guarda

model.save('model/my_model.h5')  # Se guarda con una extensión h.5
