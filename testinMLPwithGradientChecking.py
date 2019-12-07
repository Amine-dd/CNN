# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:53:38 2019

@author: Bouslimi
"""
from ANNGC import MLPGradientCheck
nn_check = MLPGradientCheck(n_output=10,n_features = X_train.shape[1],n_hidden=10,l2=0.0,l1=0.0,epochs=10,eta=0.001,alpha=0.0,decrease_const=0.0,minibatches=1,random_state=1)
nn_check.fit(X_train[:5],y_train[:5],print_progress=False)