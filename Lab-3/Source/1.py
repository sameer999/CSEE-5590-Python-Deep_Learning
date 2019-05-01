import numpy as np
import pandas as pd
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

df = pd.read_csv('crimedata.csv')
data = pd.DataFrame(df, columns=["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"])
label_col = 'medv'
print(data.describe())


x_train, x_valid, y_train, y_valid = train_test_split(data.iloc[:,0:13], data.iloc[:,13], test_size=0.3, random_state=87)
np.random.seed(155)

def model1(x_size, y_size):
    model = Sequential()
    model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(y_size))
    print(model.summary())
    keras.optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=[metrics.mae])
    return(model)

Model = model1(x_train.shape[1], 1)

Model.summary()

history = Model.fit(x_train, y_train, batch_size=64, epochs=25, shuffle=True, verbose=0, validation_data=(x_valid, y_valid), callbacks=[keras.callbacks.TensorBoard(log_dir="./1", histogram_freq=1, write_graph=True, write_images=True)])

train_score = Model.evaluate(x_train, y_train, verbose=0)
valid_score = Model.evaluate(x_valid, y_valid, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))