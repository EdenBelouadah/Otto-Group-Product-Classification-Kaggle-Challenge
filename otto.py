# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:41:58 2017

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #(setting multi_class=”multinomial”)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import os

path = "D:/mariem/Academics/master/cours/TC1/project"

def read_data(csvfile):
    df = pd.read_csv(csvfile)
    return df

if __name__=="__main__":
    #read data
    trainFile = os.path.join(path, "train.csv")
    testFile =  os.path.join(path, "test.csv")
    train_df = read_data(trainFile)
    #summary statistics of train data
    summary = train_df.describe()
    #count the number of unique levels for each feature as well as for Target 
    nb_levels = pd.DataFrame()
    cols = train_df.columns
    for j in range(train_df.shape[1]):
        l = train_df[cols[j]].unique()
        nb_levels[cols[j]] = pd.Series(len(l))
    #distribution of levels for each feature
#    for feat in cols[1:]:
#        figName = feat+'.png'
#        f = plt.figure()
#        sns.countplot(x=feat, data= train_df,
#             order = train_df[feat].value_counts().index, palette="Greens_d")
#        plt.xticks(rotation=85, fontsize = 8)
#        f.savefig(os.path.join(path, "EDA", figName))
    #study feature correlation 
    corr = train_df[cols[1:-1]].corr()
    #seperate features and target
    X = train_df[cols[1:-1]]
    y = train_df[cols[-1]]
    pca = PCA()
    pca.fit(X)
    exp_var_ratio = pca.explained_variance_ratio_
    
    #feature selection
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model1 = SelectFromModel(lsvc, prefit=True)
    idx1 = model1.get_support(indices=True)
    X_new = model1.transform(X)
    X_new.shape
    feat_selected1 = list(X.columns[idx1])
    
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    model2 = SelectFromModel(clf, prefit=True)
    idx2 = model2.get_support(indices=True)
    X_new = model2.transform(X)
    X_new.shape   
    feat_selected2 = list(X.columns[idx2])
    
    model3 = SelectKBest(mutual_info_classif, k=40).fit(X,y)
    idx3 = model3.get_support(indices=True)
    X_new = model3.transform(X) 
    X_new.shape
    feat_selected3 = list(X.columns[idx3])
    
    #spit data into train/test/validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                          test_size =0.2, random_state = 42)
    
    #try out some classification models
    models = [KNeighborsClassifier(5), RandomForestClassifier(), LinearSVC(), 
              LogisticRegression(), GradientBoostingClassifier(), MLPClassifier(), 
                                GaussianNB(), QuadraticDiscriminantAnalysis() ]
    accuracy = []
    for clf in models:
        clf.fit(X_train, y_train)
        mean_acc = clf.score(X_test, y_test)
        accuracy.append(mean_acc)
        

