# -*- coding: utf-8 -*-
"""
Created on Fri May 22 01:02:46 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset=pd.read_csv("Restaurant_Reviews.tsv", quoting=3, delimiter='\t')

#Cleaning the text
import re
import nltk
nltk.download('stopwords') #To eliminite words like the, this which has no meaning
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range (0,1000):
  review=re.sub('[^a-zA-z]'," ", dataset['Review'][i])
  review=review.lower()
  review=review.split()
  ps=PorterStemmer()
  review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
  review=" ".join(review)
  corpus.append(review)
  
#Creating bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#Splitting train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Naive bayes classification
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#Predicting test results
y_pred=classifier.predict(x_test)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)