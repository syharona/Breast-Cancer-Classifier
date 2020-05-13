# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data=load_breast_cancer()

#Organize our data
label_names = data['target_names'] #Target Names represent the name of the two columns we are trying to predict
labels = data['target'] #Target represents the values we have to predict
feature_names = data['feature_names'] #Feature names are the name of the variables that will help us predict the target
features = data['data'] #The values that will help us predict

#Turning our arrays into dataframes******
df = pd.DataFrame(features, columns=feature_names)
df['cancerStrength'] = labels

#Sklearn module for splitting the data


#Actually splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
#Train represents our features for training
#Test represents our features for testing
#Random State helps us have a seed so everytime we split the data using "42" we get the same random values
#The test size represent 33% of our data


#We initialize our classifier 
gnb = GaussianNB()

#Training our model
model = gnb.fit(train, train_labels)
#Train represents our features
#Train_labels represents our labels

#Making predictions using our test set
preds = gnb.predict(test) #We use our test set to predict labels we will later see the models accuracy by comparing it to the actual data
print(preds)

#Using an accuracy score to see how well our model performed
print(accuracy_score(test_labels, preds))