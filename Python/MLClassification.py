# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:24:04 2019

@author: Cüneyt ÖZDEMİR cuneytozdemir33@gmail.com
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import pandas as pd

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Linear Discriminant Analysis(LDA)": LinearDiscriminantAnalysis(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Naive Bayes": GaussianNB(),
    #"K Means": KMeans(n_clusters=3, random_state=0),
    "Neural Net": MLPClassifier(alpha = 1),
    "Lineer Svc":LinearSVC(C=10,max_iter=1000)
    
    #"AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process": GaussianProcessClassifier()
    }

def batch_classify(x_train, y_train, x_test, y_test, no_classifiers, verbose = True):
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:10]:
        classifier.fit(x_train, y_train)        
        pcs=metrics.precision_score(y_test,classifier.predict(x_test))  #Doğruluk ,average='binary'
        fcs=metrics.f1_score(y_test,classifier.predict(x_test))  #Doğruluk
        acs=metrics.accuracy_score(y_test,classifier.predict(x_test))  #Doğruluk
        
        dict_models[classifier_name] = {'model': classifier, 'precision_score': round(pcs,4), 'f1_score': round(fcs,4), 'Accuracy_Score':round(acs,4)}
    return dict_models
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['precision_score'] for key in cls]
    training_s = [dict_models[key]['f1_score'] for key in cls]
    acs_t = [dict_models[key]['Accuracy_Score'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'precision_score', 'f1_score', 'Accuracy_Score'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'precision_score'] = training_s[ii]
        df_.loc[ii, 'f1_score'] = test_s[ii]
        df_.loc[ii, 'Accuracy_Score'] = acs_t[ii]
    
    return df_
def classification(df):
    df = df.fillna(df.median(axis=0))
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    #verilerin egitim ve test icin bolunmesi
    from sklearn.model_selection  import train_test_split
    x_train, x_test,y_train,y_test = train_test_split(np.nan_to_num(X),Y,test_size = 0.2,random_state=123)
    #train_test_split(X,Y,test_size=0.20, random_state=0)    
    x_test=pd.DataFrame(x_test)
    y_test=pd.DataFrame(y_test)
    x_train=pd.DataFrame(x_train)
    y_train=pd.DataFrame(y_train)
    
    x_test.fillna(x_test.mean(), inplace=True)
    x_train.fillna(x_train.mean(), inplace=True)
    y_train.fillna(y_train.mean(), inplace=True)
    y_test.fillna(y_test.mean(), inplace=True)
    
    
    dict_models = batch_classify(x_train, y_train, x_test, y_test, 10)
    sonuclar=pd.DataFrame(display_dict_models(dict_models))
    return sonuclar
