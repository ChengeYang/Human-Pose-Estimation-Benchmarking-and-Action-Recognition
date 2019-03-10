import argparse
import pandas as pd
import numpy as np
import mylib.data_preprocessing as dpp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU


if __name__ == "__main__":
    # Read dataset from command line
    key_word = "--dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(key_word, required=False, default='../data/skeleton_raw.csv')
    input = parser.parse_args().dataset

    # Loading training data
    try:
        raw_data = pd.read_csv(input, header=0)
    except:
        print("Dataset not exists.")
    # X: input, Y: output
    dataset = raw_data.values
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]

    # Data pre-processing
    # X = dpp.head_reference(X)
    X_pp = []
    for i in range(len(X)):
        X_pp.append(dpp.pose_normalization(X[i]))
    X_pp = np.array(X_pp)

    # Encoder the class label to number
    # Converts a class vector (integers) to binary class matrix
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(Y)
    matrix_Y = np_utils.to_categorical(encoder_Y)
    print(Y[0], ": ", encoder_Y[0])
    print(Y[650], ": ", encoder_Y[650])
    print(Y[1300], ": ", encoder_Y[1300])
    print(Y[1950], ": ", encoder_Y[1950])
    print(Y[2600], ": ", encoder_Y[2600])

    # Split into training and testing data
    # random_state:
    X_train, X_test, Y_train, Y_test = train_test_split(X_pp, matrix_Y, test_size=0.1, random_state=42)

    # Build DNN model with keras
    model = Sequential()
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='softmax'))

    # Training
    # optimiser: Adam with learning rate 0.0001
    # loss: categorical_crossentropy for the matrix form matrix_Y
    # metrics: accuracy is evaluated for the model
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # batch_size: number of samples per gradient update
    # epochs: how many times to pass through the whole training set
    # verbose: show one line for every completed epoch
    model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=2, validation_data=(X_test, Y_test))

    # Save the trained model
    model.save('../model/action_recognition.h5')
