from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pickle


def predictions(X_train, X_validation, Y_train, Y_validation , best_model):
    best_model.fit(X_train,Y_train)
    
    prediction = best_model.predict(X_validation)

    print(accuracy_score(Y_validation, prediction))
    print(confusion_matrix(Y_validation, prediction))
    print(classification_report(Y_validation, prediction))

    X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
    #Prediction of the species from the input vector
    check_prediction = best_model.predict(X_new)
    print("Prediction of Species: {}".format(check_prediction))

    # with open('SVM.pickle', 'wb') as f:
    # pickle.dump(svn, f)

