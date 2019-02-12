# -*- coding: utf-8 -*-

import dataset as Dataset
import label
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
import numpy as np
import matplotlib.pyplot as plt

xTrain, yTrain = Dataset.loadData()

model = Sequential()
model.add(Embedding(100,32, input_length=Dataset.vocabSize))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(149, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xTrain, yTrain, validation_data=(xTrain, yTrain),batch_size=10, epochs=50)
#predY = np.argmax(model.predict(xTrain), axis=1)

#input Test 
sym = [['malaise', 'debilitation', 'symptom aggravating factors', 'chill', 'uncoordination', 'fever', 'pleuritic pain', 'snuffle', 'throat sore','scleral icterus', 'nasal flaring', 'dysuria, lip smacking', 'headache', 'sneeze', 'snore', 'green sputum', 'shortness of breath', 'distress respiratory', 'blackout', 'extreme exhaustion']]

token = Dataset.tokenizer
token.fit_on_texts(sym)
xTest = token.texts_to_matrix(sym, mode='count')
predY = np.argmax(model.predict(xTest), axis=1)
print predY, label.diseaseClasses[int(predY)]

#for i in range (len(predY)):
#    print predY[i], label.diseaseClasses[predY[i]]


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
