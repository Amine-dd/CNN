# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:03:29 2019

@author: pc
"""


from Fresh import NeuralNetMLP

nn = NeuralNetMLP(n_output=10,n_features = X_train.shape[1],n_hidden=50,l2=0.1,l1=0.0,epochs=1000,eta=0.001,alpha=0.001,decrease_const=0.00001,shuffle=True,minibatches=50,random_state=1)
nn.fit(X_train,y_train,print_progress=True)
