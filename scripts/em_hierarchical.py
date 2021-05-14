from itertools import count
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import *

import seaborn as sns
from tqdm import tqdm

import scipy as sp

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error

from xgboost import plot_importance
import xgboost as xgb
from xgboost import XGBClassifier

"""
    Implement EM with XGBoost
    @param K: number of submarket
    @param X: the dataset, X has to be arranged such that 
                the first n_cont columns have continuous (float) values
                the next n_loc columns contain values about the data point locations (usually longitude, latitude)
                the next n_bool columns have boolean (0,1) values
                the last n_int columns have integer values 
    @param n_cont: the number of continuous features
    @param n_bool: the number of boolean features
    @param n_int: the number of integer features
    @param n_loc: the number of location features
"""
def em(K, X, y, n_cont, n_bool, n_int, n_loc=2):
    n_homes, n_features = X.shape
    assert(n_loc + n_cont + n_bool + n_int == n_features)

    X_cont = X[:,:n_cont]
    X_loc = X[:,n_cont:(n_cont + n_loc)]
    X_bool = X[:,(n_cont+n_loc):(n_cont+n_loc+n_bool)]
    X_int = X[:,(n_cont+n_loc+n_bool):]
    X_feat = X[:,[i for i in range(n_features) if i not in [n_cont,n_cont+1]]]

    underflow_scaling = 1e10

    # Initialization

    init_clustering = KMeans(n_clusters=K, random_state=0).fit(X)
    prior_var = 30
    reg = 1 # Regularization strength (inverse)

    mu_init = init_clustering.cluster_centers_[:,:n_cont]
    muloc_init = init_clustering.cluster_centers_[:,n_cont:(n_cont+n_loc)]
    sigma_init = np.array([prior_var*np.eye(n_cont) for _ in range(K)])
    sigmaloc_init = np.array([prior_var*np.eye(n_loc) for _ in range(K)])
    p_init = np.maximum(init_clustering.cluster_centers_[:,(n_cont+n_loc):(n_cont+n_loc+n_bool)],0)
    lam_init = init_clustering.cluster_centers_[:,(n_cont+n_loc+n_bool):]
    pi_init = np.ones(K)/K
    # f_init = [LogisticRegression(C=reg).fit(X_feat,y) for _ in range(K)]
    f_init = [XGBClassifier(eval_metric='logloss').fit(X_feat,y) for _ in range(K)]

    # EM Implementation

    muloc = muloc_init
    sigmaloc = sigmaloc_init
    mu = mu_init
    sigma = sigma_init
    p = p_init
    lam = lam_init
    pi = pi_init
    f = f_init

    max_iter = 1000
    store_freq = 2000

    params = {int(i*store_freq): {} for i in range(int(max_iter/store_freq)+1)}

    for i in tqdm(range(max_iter),desc='Fitting...'):

        # E-step

        r_unnormalized = np.array([(underflow_scaling * pi[k] * 
                                    sp.stats.multivariate_normal(mean=muloc[k],cov=sigmaloc[k]).pdf(X_loc) *
                                    sp.stats.multivariate_normal(mean=mu[k],cov=sigma[k]).pdf(X_cont) *
                                    sp.stats.bernoulli(p[k]).pmf(X_bool).prod(axis=1) *
                                    sp.stats.poisson(lam[k]).pmf(X_int).prod(axis=1) *
                                    sp.stats.bernoulli(f[k].predict_proba(X_feat)[:,1]).pmf(y))
                                for k in range(K)]).T
        r_unnormalized[np.where(~r_unnormalized.any(axis=1))[0]] = 1/K
        r = r_unnormalized / r_unnormalized.sum(axis=1).reshape((-1,1))

        if i % store_freq == 0:
            params[i] = {'muloc': muloc,
                        'sigmaloc': sigmaloc,
                        'mu': mu,
                        'sigma': sigma,
                        'p': p,
                        'lambda': lam,
                        'pi': pi,
                        'f': f}

        # M-step

    #     f = [LogisticRegression(C=reg, max_iter=1000).fit(X_feat,y,sample_weight=r[:,k]) for k in range(K)]
        f = [XGBClassifier(eval_metric='logloss').fit(X_feat,y,sample_weight=r[:,k]) for k in range(K)]
        pi = r.sum(axis=0)/n_homes
        means = np.array([np.array([r[n,k]*X[n] for n in range(n_homes)]).sum(axis=0) for k in range(K)]) / r.sum(axis=0).reshape((-1,1))
        mu, muloc, p, lam = means[:,:n_cont], means[:,n_cont:(n_cont+n_loc)], means[:,(n_cont+n_loc):(n_cont+n_loc+n_bool)], means[:,(n_cont+n_loc+n_bool):]
        sigma = np.array([np.array([r[n,k]*np.matmul((X_cont[n]-mu[k]).reshape((-1,1)),(X_cont[n]-mu[k]).reshape((1,-1))) for n in range(n_homes)]).sum(axis=0) for k in range(K)])  / r.sum(axis=0).reshape((-1,1,1))
        sigmaloc = np.array([np.array([r[n,k]*np.matmul((X_loc[n]-muloc[k]).reshape((-1,1)),(X_loc[n]-muloc[k]).reshape((1,-1))) for n in range(n_homes)]).sum(axis=0) for k in range(K)])  / r.sum(axis=0).reshape((-1,1,1))
        
    params[max_iter] = {'muloc': muloc,
                        'sigmaloc': sigmaloc,
                        'mu': mu,
                        'sigma': sigma,
                        'p': p,
                        'lambda': lam,
                        'pi': pi,
                        'f': f}

    r_unnormalized = np.array([(underflow_scaling * pi[k] * 
                                sp.stats.multivariate_normal(mean=muloc[k],cov=sigmaloc[k]).pdf(X_loc) *
                                sp.stats.multivariate_normal(mean=mu[k],cov=sigma[k]).pdf(X_cont) *
                                sp.stats.bernoulli(p[k]).pmf(X_bool).prod(axis=1) *
                                sp.stats.poisson(lam[k]).pmf(X_int).prod(axis=1) *
                                sp.stats.bernoulli(f[k].predict_proba(X_feat)[:,1]).pmf(y))
                                for k in range(K)]).T
    r_unnormalized[np.where(~r_unnormalized.any(axis=1))[0]] = 1/K
    r = r_unnormalized / r_unnormalized.sum(axis=1).reshape((-1,1))

    submarket = np.argmax(r,axis=1)

    return submarket


def submarket_em_eval(X, y, sub, K):
    
    train_acc_agg = 0
    train_size = 0
    test_acc_agg = 0
    test_size = 0

    test = np.array([])
    pred = np.array([])

    exp = np.zeros(K)
    sales = np.zeros(K)

    for k in range(K):
        
        X_k = X[sub == k]
        Y_k = y[sub == k]
        
        if X_k.shape[0] == 0:
            continue
        # Ensure that sub-market has both classifications
        labels = np.unique(Y_k)
        if len(labels) == 1:
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(X_k, Y_k, test_size=0.3)

        model_k = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)

        train_acc = model_k.score(X_train, y_train)
        test_acc = model_k.score(X_test, y_test)

        pred_k = model_k.predict(X_test)

        exp[k] = sum(model_k.predict_proba(X_test)[:,1])
        sales[k] = sum(y_test)
        test = np.append(test,y_test)
        pred = np.append(pred,pred_k)
        auc = roc_auc_score(y_test, pred_k) 

        print("Sub-Market #{} Demand Prediction".format(k+1))
        print("Number of Homes: {}".format(len(Y_k)))
        print("Training Accuracy: {:.4f}%".format(train_acc*100))
        print("Testing Accuracy: {:.4f}%".format(test_acc*100))
        print("AUC: {:.6f}".format(auc))
        print("Expected Number of Sales: {:.6f}".format(exp[k]))
        print("Actual Number of Sales: {}\n".format(sales[k]))

        train_size += len(y_train)
        train_acc_agg += train_acc * len(y_train)
        test_size += len(y_test)
        test_acc_agg += test_acc * len(y_test)
    
    auc_agg = roc_auc_score(test, pred)
    r2_agg = r2_score(sales,exp)
    mse_agg = mean_squared_error(sales,exp)

    print('Marketwide Demand Prediction')
    print("Number of Homes: {}".format(len(y)))
    print("Training Accuracy: {:.4f}%".format(train_acc_agg*100/train_size))
    print("Testing Accuracy: {:.4f}%".format(test_acc_agg*100/test_size))
    print("AUC: {:.6f}".format(auc_agg))
    print("Expected Number of Sales: {:.6f}".format(sum(exp)))
    print("Actual Number of Sales: {}".format(sum(sales)))
    print("R2 Score (Submarket EV): {:.6f}".format(r2_agg))
    print("MSE (Submarket EV): {:.6f}".format(mse_agg))