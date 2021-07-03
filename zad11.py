import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

num_classes=3

features, labels = load_iris(return_X_y=True)
features_train,features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, shuffle=True)
labels_train=keras.utils.to_categorical(labels_train, num_classes)
labels_test=keras.utils.to_categorical(labels_test, num_classes)

model=keras.Sequential([
    keras.layers.Dense(10,activation=tf.nn.relu, input_shape=(4,)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(3)
])

print(model.summary())

batch_size = 8
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(features_train, labels_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(features_test, labels_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])