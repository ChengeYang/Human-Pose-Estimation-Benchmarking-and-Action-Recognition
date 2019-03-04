import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


def head_reference(X):
    for i in range(len(X)):
        for j in range(1, int(len(X[i])/2)):
            X[i, j*2] = X[i, j*2] - X[i, 0]
            X[i, j*2+1] = X[i, j*2+1] - X[i, 1]
    return X


if __name__ == "__main__":
    # Loading training data
    # Four class: stand, walk, squat, wave, 900 training frames each class
    # X: input, Y: output
    raw_data = pd.read_csv("data.csv", header=0)
    dataset = raw_data.values
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]
    X = head_reference(X)

    # Encoder the class label to number
    # Converts a class vector (integers) to binary class matrix
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(Y)
    matrix_Y = np_utils.to_categorical(encoder_Y)
    print(Y[0], ": ", encoder_Y[0])
    print(Y[900], ": ", encoder_Y[900])
    print(Y[1800], ": ", encoder_Y[1800])
    print(Y[2700], ": ", encoder_Y[2700])

    # Split into training and testing data
    # random_state:
    X_train, X_test, Y_train, Y_test = train_test_split(X, matrix_Y, test_size=0.1, random_state=42)

    # Build DNN model with keras
    model = Sequential()
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=4, activation='softmax'))

    # Training
    # optimiser: Adam with learning rate 0.0001
    # loss: categorical_crossentropy for the matrix form matrix_Y
    # metrics: accuracy is evaluated for the model
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # batch_size: number of samples per gradient update
    # epochs: how many times to pass through the whole training set
    # verbose: show one line for every completed epoch
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=2, validation_data=(X_test, Y_test))

    # Save the trained model
    model.save('action_recognition.h5')
