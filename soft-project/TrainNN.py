import keras.datasets as data
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.python.keras.optimizers import SGD

from methods import invert, erode, dilate, scale_to_range_n_flaten, image_bin

(train_list, train_actual), (test_list, test_actual) = data.mnist.load_data()

print("Ocekivano primer", train_actual[0])
print("Slika primer shape", train_list[0].shape)
plt.imshow(train_list[0])
plt.show()

#priprema za ulaz u neuronsku
train_prepared = []
train_conf_prep = []

for x in train_list:
    img = invert(x)
    img = erode(dilate(img))
    img = image_bin(img)
    img = scale_to_range_n_flaten(img)
    train_prepared.append(img)

train_prepared = np.array(train_prepared, np.float32)

#for x in train_actual:
#    train_conf_prep.append(x)

test_prepared = []
test_conf_prep = []

for x in test_list:
    img = invert(x)
    img = erode(dilate(img))
    img = image_bin(img)
    img = scale_to_range_n_flaten(img)
    test_prepared.append(img)

test_prepared = np.array(test_prepared, np.float32)

#for x in test_actual:
#    test_conf_prep.append(x)

print(train_prepared.__sizeof__())
print(train_conf_prep.__sizeof__())
print(test_prepared.__sizeof__())
print(test_conf_prep.__sizeof__())


#Kreiramo 10 klasa podataka
train_ac = np_utils.to_categorical(train_actual, 10)
test_ac = np_utils.to_categorical(test_actual, 10)


ann = Sequential()

ann.add(Dense(512, input_shape=(784,), activation='tanh'))
ann.add(Dropout(0.2))
ann.add(Dense(512, activation='tanh'))
ann.add(Dropout(0.2))
ann.add(Dense(512, activation='tanh'))
ann.add(Dropout(0.2))
ann.add(Dense(512, activation='tanh'))
ann.add(Dropout(0.2))
ann.add(Dense(10, activation='softmax'))

#sgd = SGD(lr=0.01, momentum=0.9)

ann.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

ann.fit(train_prepared, train_ac, epochs=10, batch_size=256, verbose=1, validation_data=(test_prepared, test_ac))

ann.save('neuronska.h5')


