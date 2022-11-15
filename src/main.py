# load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.svm import SVC
# reading dataset
import pickle
from dataload.load_dataset import load_data

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# url = "src\dataload\iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# load data from url
df = load_data(url, names)
print(df.info())
print(df.describe())

# # validation
from data_validation.validate_data import validation
X_train, X_validation, Y_train, Y_validation = validation(df)

# visualisation
from visualization.data_visualisation import data_visual
# 
data_visual(df)

from training.model_training import build_models
best_model = build_models(X_train,Y_train)

from prediction.data_prediction import predictions
predictions(X_train, X_validation, Y_train, Y_validation,best_model)