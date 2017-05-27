# -*- coding: utf-8 -*-
"""
@author: tj27
"""

from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#print(train_data.head())

def data_preprocessing_1(df):
    df.Age = df.Age.fillna(-0.5)
    df.Cabin = df.Cabin.fillna('N')
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1,0,5,12,18,25,35,60,120)
    groups = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', \
              'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels = groups)
    df.Age = categories
    
    df = df.drop(['Ticket','Embarked'], axis = 1)
    return df

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Name']
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        enc_obj = preprocessing.LabelEncoder()
        enc_obj = enc_obj.fit(df_combined[feature])
        df_train[feature] = enc_obj.transform(df_train[feature])
        df_test[feature] = enc_obj.transform(df_test[feature])
        
    return df_train, df_test
    
train_data = data_preprocessing_1(train_data)
test_data = data_preprocessing_1(test_data)

train_data, test_data = encode_features(train_data, test_data)
#print(train_data.head())

x_total = train_data.drop(['Survived','PassengerId'], axis = 1)
y_total = train_data['Survived']

no_of_test = 0.15
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, \
                                                    test_size = no_of_test, \
                                                    random_state = 23 )
                                                    
svc = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.1)
svc.fit(x_train, y_train)
predictions = svc.predict(x_test)
print(accuracy_score(y_test, predictions))

ids = test_data['PassengerId']
predictions = svc.predict(test_data.drop('PassengerId', axis=1))
output = pd.DataFrame({'PassengerId' : ids, 'Survived': predictions})
print (output.head())
output.to_csv('titanic-predictions.csv', index = 0)