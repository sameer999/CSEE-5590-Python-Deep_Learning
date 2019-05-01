import numpy as np
import pandas as pd
import keras_metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import regularizers

df = pd.read_csv("heart.csv", sep=',')
df.astype(float)

# Normalize values to range [0:1]
df /= df.max()

# split data into features & target columns
x= df.drop(columns = 'target')
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


# Creating the model
model = Sequential()
model.add(Dense(1024, input_shape=(13,), kernel_regularizer = regularizers.l2( 0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

# train the model
from keras.callbacks import TensorBoard
cb = TensorBoard(log_dir='./2', histogram_freq=1, write_graph=True, write_grads=True, batch_size=256, write_images=True)
history = model.fit(x_train, y_train,batch_size=256, epochs=100,verbose=0, validation_split=0.25,callbacks=[cb])

# make prediction
y_pred = model.predict_classes(x_test)

score = model.evaluate(x_test, y_test, verbose=0)

print('Loss:', score[0])
print('Accuracy:', score[1])
print('Precision:', score[2])
print('Recall:', score[3])


model.save('heart_disease_predict.h5')
