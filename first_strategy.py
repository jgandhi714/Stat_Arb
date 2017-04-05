# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:47:04 2017

@author: javes
"""

import stat_arb_functions as saf
import pandas as pd
import datetime
from statsmodels.tsa.stattools import adfuller as adf
import numpy as np

start_training = datetime.date.today() - datetime.timedelta(366*2)
end_training = datetime.date.today() - datetime.timedelta(366)
optional_str_training = "training"

EEM_close_training = saf.get_security_data("EEM", start_training, end_training, optional_str_training)[['Date', 'Adj Close']]
EEM_close_training = EEM_close_training.set_index('Date')
IGE_close_training = saf.get_security_data("IGE", start_training, end_training, optional_str_training)[['Date', 'Adj Close']]
IGE_close_training = IGE_close_training.set_index('Date')

residuals = saf.get_residuals(EEM_close_training['Adj Close'], IGE_close_training['Adj Close'])
adf = adf(residuals)

long_std_entry = -1.5
#long_thresh_entry = long_std_entry*np.std(residuals)
short_std_entry = 1.5
#short_thresh_entry = short_std_entry*np.std(residuals) 

long_std_exit = 0.75
long_thresh_exit = long_std_exit * np.std(residuals)
short_std_exit = -0.75
short_thresh_entry = short_std_exit * np.std(residuals)
#load test data
start_test = datetime.date.today() - datetime.timedelta(365)
end_test = datetime.date.today() - datetime.timedelta(1)
optional_str_test = "test"

EEM_close_test = saf.get_security_data("EEM", start_test, end_test, optional_str_test)[['Date', 'Adj Close']]
EEM_close_test = EEM_close_test.set_index('Date')
IGE_close_test = saf.get_security_data("IGE", start_test, end_test, optional_str_test)[['Date', 'Adj Close']]
IGE_close_test = IGE_close_test.set_index('Date')


EEM = saf.get_security_data("EEM", start_training, end_training, optional_str_training)
EEM['log adj close'] = np.log(EEM['Adj Close'])
EEM.set_index('Date')
IGE = saf.get_security_data("IGE", start_training, end_training, optional_str_training)
IGE['log adj close'] = np.log(IGE['Adj Close'])
IGE.set_index('Date')



EEM.columns = ["EEM " + str(col) for col in EEM.columns]
EEM = EEM.rename(columns = {'EEM Date':'Date'})
IGE.columns = ["IGE " + str(col) for col in IGE.columns]
IGE = IGE.rename(columns = {'IGE Date':'Date'})

data = EEM.merge(IGE)
data['IGE Preds'] = saf.get_orthogonal_regression(data['EEM log adj close'], data['IGE log adj close']).y
data['Spread'] = data['IGE Preds'] - data['IGE log adj close']
data['Spread_STDEV'] = (data['Spread'] - np.mean(data['Spread']))/np.std(data['Spread'])
data['Cum_PNL'] = 0
data['Entry_Signal'] = 0
data['Exit_Signal'] = 0
data['Position'] = 0

data.loc[data.Spread_STDEV <= long_std_entry, 'Signal'] = 1
data.loc[data.Spread_STDEV >= short_std_entry, 'Signal'] = -1



