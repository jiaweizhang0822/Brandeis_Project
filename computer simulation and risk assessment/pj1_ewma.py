#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:00:08 2019

@author: jiawei
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import function as f 
import seaborn as sns


price = pd.read_excel('ICF_data.xlsx', sheet_name='Price',header=0, index_col = 0)
#original weights--28 stocks
wts_allstock = np.array([8.57, 7.8, 7.68, 6.58, 5.8, 5.21, 4.92, 4.9, 4.36, 4.11, 3.85, 3.64, 3.38, 
                         2.72, 2.49, 2.3, 2.21, 2.06, 2.04, 1.95, 1.82, 1.81, 1.57, 1.36, 1.35, 
                         1.16, 1.16, 0.85])/100
total_stock_ticker=['AMT','PLD','SPG','EQIX','PSA','WELL','EQR','AVB','DLR','VTR','O','BXP','ESS',
         'ARE','HST','EXR','UDR','VNO','REG','DRE','ELS','FRT','NNN','KRC','SLG','ACC',
         'DEI','PK','AMH','HIW']


ret=price.pct_change().dropna()
total_day, total_num_stock1 = ret.shape
#demean ret
ret = ret - ret.mean()
#choose top 15 weighted stock as our new index
total_num_stock=15
newetf_ticker=total_stock_ticker[0:total_num_stock]
ret_newetf=ret.iloc[:,0:total_num_stock]
#recalculat weights
wts_final=wts_allstock[0:total_num_stock]/(np.sum(wts_allstock[0:total_num_stock]))


month=11*12
day_in_month=int(len(ret)/month)
#set train data set 5 years long and test 3 years
len_train_month=60
len_test_month=36
period=month-len_train_month-len_test_month+1
#3 year var, original weights to calculate -- realized te


wts_n=np.zeros((period,total_num_stock,total_num_stock))
#active wts
wts_active_n=np.zeros((period,total_num_stock,total_num_stock))
TE=np.zeros((period,total_num_stock))

#get_wts_for_n_chopse_1to15=False

#EWMA method
for i in range(period):
    ret_train=ret_newetf.iloc[0:21*(60+i),]
# var_ewma calculation of the covraiance using the function from module function.py
    lamda = 0.94
    var_ewma = f.ewma_cov(ret_train, lamda)
    cov_end = var_ewma[-1,:]
    cov_end_annual = cov_end * 252
    for n_choose in range(1,total_num_stock+1):
        if n_choose==11:
            w,y=f.te_opt_gekko(total_num_stock,n_choose,cov_end_annual,wts_final)
            wts_n[i][n_choose-1]=np.array(w).T
            wts_active_n[i][n_choose-1]=np.array(w).T-wts_final
            TE[i][n_choose-1]=f.tracking_error(wts_active_n[i][n_choose-1],cov_end_annual)

with pd.ExcelWriter('wts.xlsx') as writer:  # doctest: +SKIP
    for i in range(period):
        pd.DataFrame(wts_n[i]).to_excel(writer, sheet_name='period{}'.format(i))
with pd.ExcelWriter('wts_active.xlsx') as writer:  # doctest: +SKIP
    for i in range(period):
        pd.DataFrame(wts_active_n[i]).to_excel(writer, sheet_name='period{}'.format(i))
pd.DataFrame(TE).to_excel('TE.xlsx')




wts_active1=np.zeros((period,total_num_stock))
for i in range(period):
    wts_active1[i]=wts_active_n[i][10]

realized_te=f.realized_te(ret_newetf,wts_active1,period)


price1 = pd.read_excel('ICF_data.xlsx', sheet_name='Price',header=0)
TE1=np.zeros(period)
for i in range(period):
    TE1[i]=TE[i][10]
figure_count = 1
plt.figure(figure_count,figsize=(18,10))
date=price1.iloc[:,0]
index = np.array(21*(96+np.arange(period)))

te_train = plt.plot(date[index], realized_te*10000,color='black',label='Test data'
                    ,linewidth=4)
te_test = plt.plot(date[index], TE1*10000,color='orange',label='Train data',
                   linewidth=4)
     
plt.xlabel('Period',fontsize=20)
plt.ylabel('TE (bps)',fontsize=20)
plt.ylim(0,400)
plt.legend(fontsize = 30)      
plt.xticks(fontsize=15)   
plt.yticks(fontsize=15)   
plt.show()


#------plot TE as a function of number of stocks -------------
figure_count = figure_count+1
plt.figure(figure_count,figsize=(18,10))
plt.plot(range(1,total_num_stock+1), TE[15]*10000, 'black')
plt.xlabel('Number of stocks in ETF',fontsize=20)
plt.ylabel('Optimized Tracking Error (bps)',fontsize=20)
plt.xticks(fontsize=15)   
plt.yticks(fontsize=15)   
plt.title('ICF ETF',fontsize=20)


for i in range(period-31):
    figure_count+=1
    # ---  create plot of weights fund vs benchmark
    plt.figure(figure_count,figsize=(9,6))
    index = np.arange(len(wts_final))
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, wts_final, bar_width,
                     alpha=opacity,
                     color='black',
                     label='Index Weight')
     
    rects2 = plt.bar(index + bar_width, wts_n[i][10], bar_width,
                     alpha=opacity,
                     color='orange',
                     label='ETF fund Weight')
     
    plt.xlabel('Ticker')
    plt.ylabel('Weights')
    plt.xticks(index + bar_width, newetf_ticker)
    plt.legend()         
    plt.tight_layout()
    plt.show()
    
#wts heat plot
wts_n_for_plot=pd.DataFrame(wts_n[15])
new_index=np.arange(1,16)
wts_n_for_plot.index= new_index
wts_n_for_plot.columns=newetf_ticker
figure_count+=1
plt.figure(figure_count,figsize=(18,10))
sns.heatmap(wts_n_for_plot,annot=True,cmap="YlGnBu",linewidths=.5,vmin=0, vmax=0.3)    
    

#ret plot
figure_count = figure_count+1
plt.figure(figure_count,figsize=(6,4))
date=price1.iloc[:,0]
date1=np.array(date.loc[date!=date[0]])
wts_final1=wts_final.reshape(len(wts_final),1)
index_ret=np.dot(ret_newetf,wts_final1)
index_ret_plot = plt.plot(date1, index_ret,color='black',label='Index')
     
#te_test = plt.plot(date[index], TE1*10000,color='orange',label='New Etf')
plt.xlabel('Period')
plt.ylabel('TE')
plt.legend()         
plt.show()

price_newetf=price.iloc[:,0:total_num_stock]
index_price=np.dot(price_newetf,wts_final1)
figure_count = figure_count+1
plt.figure(figure_count)
plt.plot(price1.iloc[:,0],index_price,color='#21D150')
plt.plot(price['AVB'])
plt.show()