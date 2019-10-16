import os
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn import metrics
from numpy import inf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import *
#%matplotlib inline

dataset = pd.read_csv("C:/Users/Heather Louise/Documents/4th year CSL/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv")
datasetTest = pd.read_csv("C:/Users/Heather Louise/Documents/4th year CSL/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv")

x = dataset.drop("Income in EUR", axis =1)
y = dataset["Income in EUR"].values

#X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.7)
X_test_true = datasetTest.drop('Income', axis=1) 

num_features = ['Year of Record', 'Age', 'Size of City', 'Wears Glasses', 'Body Height [cm]']
num_transformer = Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())])

cat_features = ['Gender', 'Country', 'Profession', 'University Degree', 'Hair Color']
cat_transformer = Pipeline(steps=[('onehot', TargetEncoder()),('imputer', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])

hello = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', AdaBoostRegressor(base_estimator=RandomForestRegressor()))])

hello.fit(x, y)

y_predict = hello.predict(X_test_true)
#y_predict_true = hello.predict(X_test_true)

#print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))


dataset_sub = pd.read_csv("C:/Users/Heather Louise/Documents/4th year CSL/tcdml1920-income-ind/sub1/tcd ml 2019-20 income prediction submission file.csv")
dataset_sub['Income'] = y_predict
dataset_sub.to_csv("C:/Users/Heather Louise/Documents/4th year CSL/tcdml1920-income-ind/sub1/submission5.csv", index = False)






#print(y_predict)


