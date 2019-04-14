#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:32:39 2019

@author: jiawei
"""
from scipy import stats
import pandas_datareader.data as web
import pandas as pd  
import numpy as np
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()  #fix original function of pandas_datareader.data
import datetime 
import warnings
warnings.filterwarnings("ignore")
from pandas import ExcelWriter
import matplotlib.pyplot as plt



def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    ticker1=pdr.get_data_yahoo(ticker, start=startdate, end=enddate)
    ticker1.rename(columns=({'Adj Close':ticker}), inplace=True)                            
    return ticker1
  datas = map(getData, tickers)
  portfolio=pd.concat(datas, axis=1)
  portfolio=portfolio[tickers]
  return portfolio

tickers=['AMT','PLD','SPG','EQIX','PSA','WELL','EQR','AVB','DLR','VTR','O','BXP','ESS',
         'ARE','HST','EXR','UDR','VNO','REG','DRE','ELS','FRT','NNN','KRC','SLG','ACC',
         'DEI','PK','AMH','HIW']
start_dt = datetime.datetime(2007, 12, 26)
end_dt = datetime.datetime(2018, 12, 31)
stock_price = getDataBatch(tickers, start_dt, end_dt)
stock_price.isna().sum()
stock_price = stock_price.drop(['PK','AMH'],axis=1)
stock_price.to_excel('ICF_data.xlsx', sheet_name='Price',startrow=0, startcol=0, header=True, index=True)

