#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd #data processing, i.e reading the data from CSV file(e.g pd.read_csv)
import matplotlib.pyplot as plt #Matlab-like way of plotting
from sklearn.preprocessing import LabelEncoder


# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# read the data
df = pd.read_csv('nba_rookie_data.csv')
df

#dropping the name column 
df.drop('Name',axis=1, inplace=True)

#drop all NaN values
df.dropna(inplace=True)

# correlation metrix between all indepedent variable and the dependent vaiable
print(df.corr()['TARGET_5Yrs'].sort_values(ascending=False))

#select a single feature for logistic regression
X = df.iloc[:,[0]].values
Y = df.iloc[:,-1].values

#train test data
X_train, X_test,Y_train ,Y_test = train_test_split(X,Y, test_size = 1/3 ,random_state =0 )

#fit data
model =LogisticRegression()
model.fit(X_train,Y_train)

#predict accuracy
print('Our Accuracy is %.2f' % model.score(X_test,Y_test))

# calculate mislabed point
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))

df

#Visualize data
fig1, axes1 = plt.subplots ()
axes1.scatter(X_test, Y_test, color='blue') 
axes1.scatter(X_test, model.predict(X_test), color='red',marker='*')
axes1.scatter(X_test, model.predict_proba(X_test)[:,1], color='green', marker='.') 
axes1.set_xlabel('Games Played') 
axes1.set_ylabel( 'TARGET_5Yrs')
axes1.set_title( 'TARGET_5Yrs VS Games Played')





# In[88]:


#select multiple features for logistic Regression
X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

# Use LabelEncoder to encode categorical variables in X

label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if type(X[0, i]) == str:  # Check if the feature is categorical (assuming string type for simplicity)
        X[:, i] = label_encoder.fit_transform(X[:, i])


#train test data
X_train, X_test,Y_train ,Y_test = train_test_split(X,Y, test_size = 1/3 ,random_state =0)

#fit data
model =LogisticRegression(max_iter=2000)
model.fit(X_train,Y_train)

#predict accuracy
print('Our Accuracy is %.2f' % model.score(X_test,Y_test))

# calculate mislabed point
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))

df


# In[90]:


#select a single feature for Gaussian
X = df.iloc[:,[0]].values
Y = df.iloc[:,-1].values

#train test data
X_train, X_test,Y_train ,Y_test = train_test_split(X,Y, test_size = 1/3 ,random_state =0)

#fit data
model = GaussianNB()
model.fit(X_train,Y_train)

#predict accuracy
print('Our Accuracy is %.2f' % model.score(X_test,Y_test))

# calculate mislabed point
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))

df

#Visualize data
fig2, axes2 = plt.subplots ()
axes2.scatter(X_test, Y_test, color='blue') 
axes2.scatter(X_test, model.predict(X_test), color='red',marker='*')
axes2.scatter(X_test, model.predict_proba(X_test)[:,1], color='green', marker='.') 
axes2.set_xlabel('Games _played') 
axes2.set_ylabel( 'TARGET_5Yrs')
axes2.set_title( 'TARGET_5Yrs VS Games_played')


# In[65]:


#select several features for multiple using Gaussian
X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

#train test data
X_train, X_test,Y_train ,Y_test = train_test_split(X,Y, test_size = 1/3 ,random_state =0)

#fit data
model = GaussianNB()
model.fit(X_train,Y_train)

#predict accuracy
print('Our Accuracy is %.2f' % model.score(X_test,Y_test))

# calculate mislabed point
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (Y_test != model.predict(X_test)).sum()))

df



# In[91]:


#select a single feature for neural_network
from sklearn.neural_network import MLPClassifier
X = df.iloc[:,[0]].values
Y = df.iloc[:,-1].values

#split train test data
X_train, X_test,Y_train ,Y_test = train_test_split(X,Y, test_size = 1/3 ,random_state =0)

#setup the neural network architecture
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),
activation="tanh",random_state=0,max_iter=2000)
mlp.fit(X_train,Y_train)

#performance metrics
print('Our Accuracy is %.2f' % mlp.score(X_test,Y_test))
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (Y_test != mlp.predict(X_test)).sum()))

df


# In[100]:


#select multiple features for neural_network
X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

#split train test data
X_train, X_test,Y_train ,Y_test = train_test_split(X,Y, test_size = 1/3 ,random_state =0)

#setup the neural network architecture
mlp = MLPClassifier(hidden_layer_sizes=(),
activation="tanh",random_state=0,max_iter=2000)
mlp.fit(X_train,Y_train)

#performance metrics
print('Our Accuracy is %.2f' % mlp.score(X_test,Y_test))
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (Y_test != mlp.predict(X_test)).sum()))

df


# In[ ]:





# In[ ]:




