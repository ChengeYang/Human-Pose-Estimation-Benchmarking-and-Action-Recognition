import pandas as pd
import numpy as np
import data_preprocessing as dpp
from keras.models import load_model


if __name__ == "__main__":
    # Loading training data
    # Four class: stand, walk, squat, wave, 900 training frames each class
    # X: input, Y: output
    raw_data = pd.read_csv("data.csv", header=0)
    dataset = raw_data.values
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]

    # Data pre-processing
    # X = dpp.head_reference(X)
    X_pp = []
    for i in range(len(X)):
        X_pp.append(dpp.pos2angles(X[i]))
    X_pp = np.array(X_pp)

    # Loading model
    model = load_model('action_recognition.h5')

    # Testing
    correct_num = 0
    class_table = {0:"squat", 1:"stand", 2:"walk", 3:"wave"}
    for i in range(0, len(Y)):
        test_x = np.array(X_pp[i]).reshape(-1, len(X_pp[i]))
        test_y = Y[i]
        if test_x.size > 0:
            pred_num = np.argmax(model.predict(test_x))
            pred_class = class_table[pred_num]
            if pred_class == test_y:
                correct_num += 1
    accuracy = correct_num / len(Y) * 100
    print("Accuracy:", accuracy, "%")
