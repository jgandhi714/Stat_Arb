# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:43:13 2017

@author: javes
"""

import pandas_datareader as web
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.tools as sm2
from statsmodels.tsa.stattools import adfuller as adf
import numpy as np
from scipy import odr
import os.path

def get_data_fp(sec_string, start_date, end_date, optional_string = ""):
    "Returns file path of data for given params"
    start_str = str(start_date)
    end_str = str(end_date)
    if optional_string != "":
        file_path = "C:/Users/Javes/Documents/Python Scripts/Stat Arb/data/" + sec_string + "_from_" + start_str + "_to_" + end_str + "_" + optional_string + ".csv"
    else:
        file_path = "C:/Users/Javes/Documents/Python Scripts/Stat Arb/data/" + sec_string + "_from_" + start_str + "_to_" + end_str + ".csv"
    return file_path
def get_security_data(sec_string, start_date, end_date, optional_string = ""):
    """sec_string is the string for the security ticker you want data for
    the start and end date are in datetime format - i.e. datetime.date(2016, 3, 23)
    the optional string will be used as the last string in the file name, i.e. train data
    the function will output a dataframe with the price data for the given security and dates"""
    #first check if data already exists; don't redownload!
    file_path = get_data_fp(sec_string, start_date, end_date, optional_string)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    sec_data = web.DataReader(sec_string, 'yahoo', start_date, end_date)
    sec_data.to_csv(file_path)
    return pd.read_csv(file_path)
    
#the following function makes a chart with overlayed series
def overlayed_chart(dates, y1, y2, y1_label, y2_label):
    fig, ax1 = plt.subplots()
    t = y1.index
    s1 = y1
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(y1_label, color = 'b')
    ax2 = ax1.twinx()
    s2 = y2
    ax2.plot(t, s2, 'r-')
    ax2.set_ylabel(y2_label, color = 'r')
    plt.show()

def get_linear_regression_params(X, Y, with_constant = True):
    #takes as input logged price data
    if with_constant:
        X = sm2.add_constant(X)
    mod = sm.OLS(Y, X)
    res = mod.fit()
    return res

def get_orthogonal_regression(X, Y, with_constant = True):
    #takes as input logged price data
    #runs OLS regression to get initial estimate of parameters
    initial_guess = get_linear_regression_params(X, Y, with_constant).params
    def f(B, x):
        return B[0] + B[1]*x
    def f2(B, x):
        return B[0] * x
    if with_constant:
        linear = odr.Model(f)
    else:
        linear = odr.Model(f2)
    mydata = odr.Data(X, Y)
    myodr = odr.ODR(mydata, linear, beta0=initial_guess)
    myoutput = myodr.run()
    return myoutput

def get_predictions(X, Y):
    odr_output = get_orthogonal_regression(X, Y)
    predictions = odr_output.beta[0] + odr_output.beta[1] * X
    return predictions
def get_hedge_ratio(X, Y):
    #returns hedge ratio for x (number of units of x for each unit of y)
    odr_output = get_orthogonal_regression(X, Y)
    hedge_ratio = odr_output.beta[1]
    return hedge_ratio

def get_residuals(X, Y):
    residuals = Y - get_predictions(X, Y)
    return residuals

def adf_residuals(X, Y):
    residuals = get_residuals(X, Y)
    adf_test = adf(residuals)
    return adf_test
    
    

        
    