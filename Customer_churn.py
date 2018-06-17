# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:37:14 2018

@author: Dipankar Karmakar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier



churn_data=pd.read_csv('/home/dipankarkarmakar/Downloads/customerchurn.csv')
print(churn_data.head())
print(churn_data.dtpyes)







label = preprocessing.LabelEncoder()
churn_data['Churn']=label.fit_transform(churn_data['churn'])
churn_data['State']=label.fit_transform(churn_data['state'])
churn_data['vmp']=label.fit_transform(churn_data['voice mail plan'])
churn_data['International plan']=label.fit_transform(churn_data['international plan'])


Y=churn_data['Churn']
print(churn_data.isnull().sum().sum(),'-null values')

churn_data=churn_data.drop(['Churn','churn','phone number','state','voice mail plan','international plan','area code','number vmail messages','total night calls','State','total eve calls','vmp','total night charge','account length','total day calls',],axis=1)
print(churn_data.sample(5))













X_train,X_test,Y_train,Y_test=train_test_split(churn_data,Y,test_size=0.2, random_state=4)

scl=StandardScaler()
data_scl=scl.fit_transform(X_train)
data_pd=pd.DataFrame(data_scl,columns=X_train.columns)

scl=StandardScaler()
data_scl1=scl.fit_transform(X_test)
data_pd1=pd.DataFrame(data_scl1,columns=X_test.columns)

print(X_train.head())
print(X_test.head())

random_imp=RandomForestClassifier()
random_imp.fit(data_pd,Y_train)
print(random_imp.feature_importances_)
print(data_pd.columns)
a=data_pd.columns
b=random_imp.feature_importances_
c={}
for i,j in zip(a,b):
    c.update({i:j})
print(c)
'''sorting dictionary by its value'''
from collections import Counter
c1=Counter(c)
print(c1.most_common())



rfe=RandomForestClassifier()
rfe.fit(data_pd,Y_train)
#print(rfe.score(data_pd1,Y_test))
print(rfe.score(data_pd1,Y_test))
print(rfe.score(data_pd,Y_train))


#by heatmap we see the level of covariance,then dropping the columns by the given threshold value
#-ve value means one increasing and other one decreasing
#-ve and +ve we dont see that in it we just see the hard value
sns.heatmap(data_pd.corr(), square=False,linewidths=2, annot=True, cmap="RdBu")
corr_mat=data_pd.corr()
for i in corr_mat:
    print(i)
    for j in corr_mat:
        print(j)
        if(i==j):
            print(corr_mat[i][j],'same column')
        else:
            print(corr_mat[i][j],'different')
a=set()          
for i in corr_mat:
    for j in corr_mat:
        if (i==j):
            continue
        
        else:
            if(corr_mat[i][j]>0.2):
                a.add(i)
print(a)
            

sve=SVC()
sve.fit(data_pd,Y_train)
print(sve.score(data_pd1,Y_test))
print(sve.score(data_pd,Y_train))

grb=GradientBoostingClassifier()   
grb.fit(data_pd,Y_train)
print(grb.score(data_pd1,Y_test))
print(grb.score(data_pd,Y_train))




cor_matt=data_pd.corr()
eig_vals, eig_vecs = np.linalg.eig(cor_matt)
#print(eig_vals)
#print('sdaddddddddddddddd')
#print(eig_vecs)
'''fiting and transforming pca'''
pca=PCA(n_components=9)
train_features = pca.fit_transform(data_pd)
test_features = pca.transform(data_pd1)

sve1=SVC()
sve1.fit(train_features,Y_train)
print(sve1.score(test_features,Y_test))
print(sve1.score(train_features,Y_train))

grb1=GradientBoostingClassifier()   
grb1.fit(train_features,Y_train)
print(grb1.score(test_features,Y_test))
print(grb1.score(train_features,Y_train))

rfe1=RandomForestClassifier()
rfe1.fit(train_features,Y_train)
#print(rfe.score(data_pd1,Y_test))
print(rfe1.score(test_features,Y_test))
print(rfe1.score(train_features,Y_train))

rf = RandomForestClassifier(n_estimators=5)
rf.fit(data_pd,Y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
print(rf.score(data_pd1,Y_test))


adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
adb.fit(data_pd,Y_train)
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=5, random_state=None)

print(adb.score(data_pd1,Y_test))


print(adb.score(data_pd,Y_train))


bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg.fit(data_pd,Y_train)
BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
         bootstrap=True, bootstrap_features=False, max_features=1.0,
         max_samples=0.5, n_estimators=20, n_jobs=1, oob_score=False,
         random_state=None, verbose=0, warm_start=False)
print(bg.score(data_pd1,Y_test))

print(bg.score(data_pd,Y_train))

lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf1 = RandomForestClassifier()
svm = SVC(kernel = 'poly', degree = 2 )

evc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('rf1',rf1),('svm',svm)], voting = 'hard')

evc.fit(data_pd,Y_train)
print(evc.score(data_pd1,Y_test))

print(evc.score(data_pd,Y_train))




