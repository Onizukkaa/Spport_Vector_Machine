# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:35:43 2022

@author: Joachim ANDRE
"""
"""SVR workshop"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

dataset=pd.read_csv("Position_Salaries.csv", sep=";")
print(dataset.head)
#%%
X=dataset.iloc[:,1:2].values.astype(float)
y=dataset.iloc[:,2:3].values.astype(float)
#%%
#Feature Scalling
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)
#%%
#Fitting the SVR model to the dataset. Create yout Vectore Regressor here
from sklearn.svm import SVR
#Moste important SVR parameter is Kernel type. It can be : linear, polynomial or gaussian SVR. 
#We have a non-linear condition, so we can select polynomial or gaussian but here we select RBF( a gaussian type) Kernel

regressor=SVR(kernel="rbf" , verbose=2)
regressor.fit(X,y)

#%%
#Visualing the suport vector Regression results
plt.scatter(X,y,color="magenta")
plt.plot(X,regressor.predict(X),color="green")
plt.title("Truth or bluff(SVR)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()
#%%
#Predicting a new result

y_pred = sc_y.inverse_transform((regressor.predict(sc_X.transform(np.array([[6.5]])))))
print(y_pred)
#%%
#Visualing the Regression results(for highre resolution and smoother curve)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or bluff(SVR model rbf")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()













