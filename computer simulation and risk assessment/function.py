#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:49:08 2019

@author: jiawei
"""

import numpy as np 
from scipy import optimize
import random
from gekko import GEKKO

def realized_te(ret,w_a,period):
    w_a=np.matrix(w_a)
    rea_te=np.zeros(period)
    for i in range(period):
        cov=ret.iloc[(60+i)*21:(96+i)*21, ].cov()
        cov=np.matrix(cov)
        cov_annual=cov*252
        rea_te[i]=np.sqrt(w_a[i]*cov_annual*w_a[i].T)
    return rea_te

def tracking_error(wts_active,cov):
    wts_active=np.matrix(wts_active)
    cov=np.matrix(cov)
    TE=np.sqrt(wts_active*cov*wts_active.T)
    return TE

def ewma_cov(ret, lamda): 
    T, n = ret.shape
    ret_mat = np.matrix(ret)
    EWMA = np.zeros((T+1,n,n))
    S = ret.cov()
    S=np.matrix(S)
    EWMA[0,:] = S    
    for i in range(1, T+1):
        S = lamda * S + (1-lamda) * (ret_mat[i-1,:].T*ret_mat[i-1,:])
        EWMA[i,:] = S
    return(EWMA)

def te_opt_gekko(total_num_stock,n_choose,cov,wts_final):
    m=GEKKO()
    y=[m.Var(1,lb=0,ub=1,integer=True) 
    for i in range(total_num_stock)]
    w=[m.Var(value=1./total_num_stock,lb=0,ub=1) 
    for i in range(total_num_stock)]

    m.Equation(np.sum(y)==n_choose)
    m.Equation(np.sum(w)==1.)
    m.Equations(w[i]<=y[i] for i in range(total_num_stock))
    
    w_a=[w[i]-wts_final[i] for i in range(total_num_stock)]

    m.Obj(np.dot(np.dot(w_a,cov),w_a))
    m.options.SOLVER=1
    m.solve(disp=False)
    return w,y


def obj_te(W, W_Bench, C): 
    wts_active = W - W_Bench
    wts_active=np.matrix(wts_active)
    return np.sqrt(wts_active*C*wts_active.T)

def te_opt(W_Bench, C, c_, b_):
    # function that minimize the objective function
    n = len(W_Bench)
    # change the initial guess to help test whether we find the global optimal
    W=rand_weights(n)

    optimized = optimize.minimize(obj_te, W, ( W_Bench, C), 
                method='SLSQP', constraints=c_, bounds=b_,  
                options={'ftol':1e-10, 'maxiter': 1000000, 'disp': False})
        
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights

def te_opt_n(W_Bench, C,stock_choose):
    # function that minimize the objective function
    n = len(W_Bench)
    n_choose=len(stock_choose)
    W=np.zeros(n)
    # change the initial guess to help test whether we find the global optimal
    b_=[]
    for i in range(n):
        if i in stock_choose:
            W[i]=1./n_choose   
            b_.append((0.0,1.0))
        else:
            b_.append((0.0,0.0))

    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })

    optimized = optimize.minimize(obj_te, W, (W_Bench, C), 
                method='SLSQP', constraints=c_, bounds=b_,  
                options={'ftol':1e-10, 'maxiter': 1000000, 'disp': False})
        
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights


# create random weights 
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    np.random.seed(6666)
    k = np.random.uniform(low=0,high=1,size=n)
    return k / sum(k)

def assign(n,n_choose):
    a=np.zeros(n)
    randomnumber=random.sample(range(n), n_choose)
    a[randomnumber]=1
    return a
 
def totallist(n):
    result=[]
    for i in range(n):
        result.append(i)
    return result
    


def te_opt_n2(W_Bench, C,  n_choose):
    # function that minimize the objective function
    n = len(W_Bench)
    # change the initial guess to help test whether we find the global optimal
    W=np.zeros(2*n)
    W[0:n]=1./n
    b1_ = [(0.0, W[i+n]) for i in range(n)]
    b2_ = [(0.0, 1.0) for i in range(n,2*n)]
    b_=b1_+b2_
    c_=({'type':'eq','fun': lambda W: sum(W[n:2*n])-n_choose*1.0},
       {'type':'eq','fun': lambda W: sum(W[0:n])-1.0},
       {'type':'eq','fun': lambda W: sum(  W[n:2*n]**2*(1-W[n:2*n])**2   )}
       )

    optimized = optimize.minimize(obj_te_n, W, (n, W_Bench, C), 
                method='SLSQP', constraints=c_, bounds=b_,  
                options={'ftol':1e-8, 'maxiter': 10000, 'disp': False})
        
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights

def obj_te_n(W,n, W_Bench, C): 
    wts_active = W[0:n] - W_Bench
    wts_active=np.matrix(wts_active)
    return np.sqrt(wts_active*C*wts_active.T)


