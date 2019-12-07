#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:55:11 2019

@author: davidmadsen
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
import pandas as pd
import seaborn

path = os.path.join('~',
                    'Downloads',
                    'nfl_master_sent_merged_timeseries.csv')

nfl_ts = pd.read_csv(path)

nfl_ts['created_at'] = pd.to_datetime(nfl_ts['created_at'],
             format = '%a %b %d %H:%M:%S %z %Y')

nfl_ts_list = []

for row in nfl_ts.itertuples():
    sent_dict = eval(row[15])
    sent = float(sent_dict['compound'])
    nfl_ts_dict = {
            'created_at':row[2],
            'sent':sent}
    nfl_ts_list.append(nfl_ts_dict)
    
nfl_ts_data = pd.DataFrame(nfl_ts_list)



nfl_by_day = nfl_ts_data.resample('d', on='created_at').mean().dropna(how='all')

ax = nfl_by_day['sent'].plot()
ticklabels = nfl_by_day.index.strftime('%Y-%m-%d')
#ax.xaxis.set_major_formatter(ticker.FixedFormat(ticklabels))

plt.show()

nfl_ts_list = []

for row in nfl_ts.itertuples():
    sent_dict = eval(row[15])
    sent = float(sent_dict['compound'])
    nfl_ts_dict = {
            'created_at':row[2],
            'neg':sent_dict['neg'],
            'neu':sent_dict['neu'],
            'pos':sent_dict['pos']}
    nfl_ts_list.append(nfl_ts_dict)
    
nfl_ts_data = pd.DataFrame(nfl_ts_list)

nfl_by_day = nfl_ts_data.resample('d', on='created_at').mean().dropna(how='all')

ax = nfl_by_day['neg'].plot(label='negative')
ax.plot(nfl_by_day['neu'], label='neutral')
ax.plot(nfl_by_day['pos'], label='positive')
ticklabels = nfl_by_day.index.strftime('%Y-%m-%d')

plt.legend(loc='best')
plt.show()
#ax.xaxis.set_major_formatter(ticker.FixedFormat(ticklabels))