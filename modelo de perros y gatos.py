import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbCallback, WandbModelCheckpoint, WandbMetricsLogger
#wandb.login(key="5820488c8ea56dabae7982032d79491f27512f1d")
wandb.login()

tests = "/Users/esasm/Documents/Redes neuronale/Red Neuronal con Numpy/perroygato/DogsCats/test"
train = "/Users/esasm/Documents/Redes neuronale/Red Neuronal con Numpy/perroygato/DogsCats/train"


clases= 2
aprendizaje= 0.001
epocas= 30
trainbatch= 20000
testbatch= 5000
minibatch= 50
optimizador="adam"
opti= tf.keras.optimizers.Adam(aprendizaje, beta_1=0.9, beta_2=0.999)
neurona="relu"
filtro1=12
filtro2=48
kernel=(3,3)

wandb.init(project="Perro y gato")
wandb.config.learning_rate = aprendizaje
wandb.config.epochs = epocas
wandb.config.batch_size = minibatch
wandb.config.optimizer = opti
wandb.config.neuron= neurona
wandb.config.filtro1= filtro1
wandb.config.filtro2= filtro2
wandb.config.kernel= kernel
#wandb.config.

tam_input=(150,150,3)

gentrain = ImageDataGenerator()
train = gentrain.flow_from_directory(train,
                batch_size=minibatch,
                target_size=(32, 32),
                class_mode='binary')

gentest = ImageDataGenerator()
test = gentest.flow_from_directory(tests,
                batch_size=minibatch,
                target_size=(32, 32),
                class_mode='binary')


#Modelo#

def bloque(   x, filtros, kernel, neurona):
    n= layers.Conv2D(filtros, kernel, padding="same")(x)
    n= layers.BatchNormalization()(n)
    n= layers.Activation(neurona)(n)

    n= layers.Conv2D(filtros, kernel, padding="same")(n)
    n= layers.BatchNormalization()(n)
    n= layers.Activation(neurona)(n)

    #n= layers.MaxPooling2D(pool_size=(2,2), padding="same")(n)
    x= layers.Conv2D(filtros, (1,1), )(x)

    n= layers.add([x,n])
    n= layers.Activation(neurona)(n)
    return n

#Uso del modelo# 

inputs = layers.Input(shape=(32, 32, 3))
x = layers.Rescaling(1./255)(inputs)
x = bloque(x, filtro1, kernel, neurona)
x = bloque(x, filtro2, kernel, neurona)
x = bloque(x, filtro1, kernel, neurona)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="sigmoid")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
modelo = keras.Model(inputs, outputs)
modelo.compile(optimizer=opti,
              loss='binary_crossentropy',
              metrics=['accuracy'])

modelo.summary()

modelo.fit(train, epochs=epocas, validation_data=test, callbacks=[WandbMetricsLogger(log_freq=5),
                                                                  WandbModelCheckpoint("models")])