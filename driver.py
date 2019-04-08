#!/usr/bin/python3.6
import csv
import sys
import numpy as np
from sklearn import svm

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# coding=utf-8

from sklearn.metrics import confusion_matrix

def normalize_method(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mu)/std
    return X

def algoritms_classifications(typeAlgorithm, X, y, output_f):
    estimator  = ''
    parameters = ''
    X_train, X_test, y_train, y_test = stratified_sampling(X, y)
    X_train = normalize_method(X_train)
    X_test = normalize_method(X_test)
    if (typeAlgorithm == 1):
        name = 'svm_linear'
        parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
        estimator  = svm.SVC(kernel='linear')

    elif(typeAlgorithm ==2):
        name = 'svm_polynomial'
        parameters = {'C':[0.1, 1, 3], 'degree':[4, 5, 6], 'gamma':[0.1, 0.5]}
        estimator  = svm.SVC(kernel='poly')
        
    elif(typeAlgorithm ==3):
        name = 'svm_rbf'
        parameters = {'C': [0,1, 0,5, 1, 5, 10, 50, 100] , 'gamma':[0,1, 0,5, 1, 3, 6, 10]}
        estimator  = svm.SVC(kernel='rbf')
        
    elif(typeAlgorithm ==4):
        name = 'logistic'
        parameters = {'C' : [0,1, 0,5, 1, 5, 10, 50, 100]}
        estimator = LogisticRegression()
        
    elif(typeAlgorithm ==5):
        name = 'knn'
        parameters = {'n_neighbors': range(1,50), 'leaf_size': range(5, 61, 5)}
        estimator = KNeighborsClassifier()
        
    elif(typeAlgorithm == 6):
        name = 'decision_tree'
        parameters = { 'max_depth': range(0,50), 'min_samples_split' : range(2, 10, 1)}
        estimador = DecisionTreeClassifier()
        
    elif(typeAlgorithm == 7):
        name = 'random_forest'
        parameters = { 'max_depth': range(0,50),'min_samples_split': range(1,50)}
        estimador = RandomForestClassifier()
    
    
    
    clf = GridSearchCV(estimator, parameters, cv=5, n_jobs=10)
    clf.fit(X_train, y_train)
    
    #best_estimator = clf.best_estimator_
    #trainScoreBest = clf.best_score_
    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    
    Y_train_pred = clf.predict(X_train)
    Y_test_pred = clf.predict(X_test)
    #print(clf.best_score_)
    test_score = accuracy_score(Y_test_pred, y_test)
    best_score = accuracy_score(Y_train_pred, y_train)
    print(test_score)
    print(best_score)
    #cm = confusion_matrix(y_test, y_pred)
    
    #print(classification_report(y_true, y_pred))
    #x_predict = clf.predict(X_train)
    #y_predict = clf.predict(y_train)
    with open(output_f, 'w') as csvfile:
        fieldnames = ['name', 'best_score', 'test_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'name': name, 'best_score':best_score, 'test_score': test_score})
    
def stratified_sampling(X, y):
    stratified_data = StratifiedShuffleSplit(n_splits = 1, test_size = 0.4, train_size = 0.6, random_state = 0)
    #print(stratified_data)
    for train_index, test_index in stratified_data.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

    
if __name__ == "__main__":
    #variables
    n_algorithms = 7
    #data_input
    input_f = sys.argv[1]
    output_f = sys.argv[2]
    
    points = np.genfromtxt(input_f, delimiter=",", skip_header=1)
    
    #x = points[:,0] #A column
    #y = points[:,1] #B column
    #plt.scatter(x,y)
    #plt.show()
    X = points[:,:-1]
    y = points[:,-1]
    
    
    algoritms_classifications(2, X, y, output_f=output_f)
    i= 0
    #for i in range(0,n_algorithms):
        #algoritms_classifications(i, X_train, y_train)

    #print(clf.predict([0.58,0.76]))
    #training data_set

    
    
    
