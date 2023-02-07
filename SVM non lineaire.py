# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 09:46:51 2022

@author: utilisateur
"""

"""Work shop Open classe Room SVM non Linéaire"""


import numpy as np
import pandas as pd
#%%
data=pd.read_csv("winequality-white.csv", sep=";")

#%%
#Créer la matrice de données
X=data[data.columns[:-1]].values

#Créer le vecteur d'étiquettes
y=data["quality"].values 

#Transformer en un problème de classification binaire
y_class=np.where(y<6,0,1)
#%%
from sklearn import model_selection

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y_class,test_size=0.3)
#%%
#Standardiser les données
from sklearn import preprocessing

std_scale =preprocessing.StandardScaler().fit(X_train)
X_train_std=std_scale.transform(X_train)
X_test_std=std_scale.transform(X_test)
#%%
#SVM avec un noyau gaussien de paramètre gamma=0.01
from sklearn import svm
classifier=svm.SVC(kernel="rbf",gamma=0.01)
#Entrainer le SVM sur le jeu d'entrainement
classifier.fit(X_train_std,y_train)
#%%
#Prédire sur le jeu de test
y_test_pred=classifier.decision_function(X_test_std)
#courbre ROC
from sklearn import metrics
fpr,tpr,thr=metrics.roc_curve(y_test,y_test_pred)
#calcul air sous la courbe ROC
auc=metrics.auc(fpr,tpr)
#Créer la figure
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(6,6))
#Afficher la courbe
plt.plot(fpr,tpr,"-",lw=2,label="gamma0.01, AUC=%.2f" % auc)

#Titre
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate",fontsize=16)
plt.title("SVM ROC Curve",fontsize=16)
#Afficher la légende
plt.legend(loc="lower right",fontsize=14)
#Afficher l'image
plt.show()
#%%
#Choisir 6 valeurs pour C, entre 1e-2 et 1e3
C_range=np.logspace(-2,3,6)
#Choisir 5 valeurs pour gamme entre 1e-2 et 10
gamma_range=np.logspace(-2,1,4)

#Grille de paramètres
param_grid={"C":C_range,"gamma":gamma_range}

#Critère de selection meilleur modele
score="roc_auc"

#initialiser une recherche sur grille
grid=model_selection.GridSearchCV((svm.SVC(kernel="rbf")), 
                                  param_grid,
                                  cv=5,
                                  scoring=score,
                                  verbose=2)
#Faire tourner la grid_search
grid.fit(X_train_std,y_train)

#Afficher param optimaux
print(f"The optimal params are : {grid.best_params_} qith a score of : {grid.best_score_:.2f}")
#%%
#Prédire sur le jeu de test avec model optimisé
y_test_pred_cv=grid.decision_function(X_test_std)
#Courbe ROC
fpr_cv,tpr_cv,thr_cv=metrics.roc_curve(y_test,y_test_pred_cv)
#Calcul aire sour courbe
auc_cv=metrics.auc(fpr_cv,tpr_cv)

fig=plt.figure(figsize=(6,6))
#Afficher la courbe
plt.plot(fpr,tpr,"-",lw=2,label="gamma0.01, AUC=%.2f" % auc)
#model optimisé
plt.plot(fpr_cv,tpr_cv,"-",lw=2,label="gamma=%.1e,AUC=%.2f"%(grid.best_params_["gamma"],auc_cv))

#Titre
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate",fontsize=16)
plt.title("SVM ROC Curve",fontsize=16)
#Afficher la légende
plt.legend(loc="lower right",fontsize=14)
#Afficher l'image
plt.show()
#%%
"""Matrice de Gram"""
from sklearn import metrics
import matplotlib as mp
kmatrix=metrics.pairwise.rbf_kernel(X_train_std,gamma=0.01)

#R"duction matrice à 100 premières lignes et colonnes pour faciliter visu
kmatrix100=kmatrix[:100,:100]
#Dessiner matrice
plt.pcolor(kmatrix100,cmap=mp.cm.PuRd)
#Rajouter légende
plt.colorbar()
#Retourner l'axe des ordonnées
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.show()























