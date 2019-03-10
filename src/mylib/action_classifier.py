
import numpy as np
import sys, os
import pickle 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# My
from .feature_proc import *
from .funcs import *

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Path
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# Feiyu's for Inference -----------------------------------------------
class MyClassifier(object):
    
    def __init__(self, model_path, action_types):
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        if self.model is None:
            print("my Error: failed to load model")
            assert False

        self.action_types = action_types

    def predict(self, skeleton):

        # Preprocess data
        skeleton_input = skeleton.reshape(-1, 36)

        # Predicted label: int & string
        predicted_idx = self.model.predict(skeleton_input)[0]
        prediced_label = self.action_types[predicted_idx]

        return prediced_label

# Define classifier for training-----------------------------------------------
class MyModel(object):
    def __init__(self):
        self.init_all_models()
        
#         self.clf = self.choose_model("Nearest Neighbors")
#         self.clf = self.choose_model("Linear SVM")
#         self.clf = self.choose_model("RBF SVM")
#         self.clf = self.choose_model("Gaussian Process")
#         self.clf = self.choose_model("Decision Tree")
#         self.clf = self.choose_model("Random Forest")
        self.clf = MLPClassifier((100,100,100,100,100,100))
#         self.clf = self.choose_model("AdaBoost")
#         self.clf = self.choose_model("Naive Bayes")
#         self.clf = self.choose_model("QDA")
        

    def choose_model(self, name):
        idx = self.names.index(name)
        return self.classifiers[idx]
            
    def init_all_models(self):
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    def extract_features(self, x): # x should be a 2d np.array
        if 0: # do nothing
            return x
        else:
            x_new = []
            for i in range(x.shape[0]):
                
                # Select features
                x_body_joints_xy = x[i,:]
                
#                 x_body_joints_xy = ProcFtr.retrain_only_body_joints(x_body_joints_xy)
#                 x_body_joints_xy = ProcFtr.normalize(x_body_joints_xy)
                
                x_body_joints_xy = ProcFtr.pose_normalization(x_body_joints_xy)
            

#                 x_body_joints_angle = ProcFtr.joint_pos_2_angle(x[i,:])
                
                # save to x_new
#                 xi = np.concatenate((x_body_joints_xy, x_body_joints_angle))
#                 xi = x_body_joints_angle

                xi = x_body_joints_xy
    
                x_new.append(xi)
            return np.array(x_new)
    
    def train(self, X0, Y):
        X = self.extract_features(X0)
        self.clf.fit(X, Y)
        
    def predict(self, X0):
        X = self.extract_features(X0)
        Y_predict = self.clf.predict(X)
        return Y_predict
    
    def predict_and_evaluate(self, X_test, Y_test):
        Y_test_predict = self.predict(X_test)
        N = len(Y_test)
        n = sum( Y_test_predict == Y_test )
        accu = n / N
        print("Accuracy is ", accu)
        return accu