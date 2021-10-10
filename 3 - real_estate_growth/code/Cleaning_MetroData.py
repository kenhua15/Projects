#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 23:11:04 2021

@author: kenhua
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Metro_Data_2.csv')
df.set_index('RegionName', inplace = True)
def makenan(x):
    if x == '#DIV/0!':
        return np.nan
    else:
        return x
listdf = []
appdf = pd.DataFrame()
for column in df:
    df2 = df[column].str.strip('%')
    df2 = df2.apply(makenan)
    df2 = df2.astype('float')
    na_count = df2.isna().sum()
    if na_count < 40:
        listdf.append(df2)
        appdf = pd.concat((appdf,df2), axis = 1)
av_column = appdf.mean(axis=0)
av_column = av_column.sort_values(ascending = False)
top30 = av_column.index[1:30]

top_30_df = pd.DataFrame()
for i in top30:
    top_30_df = pd.concat((top_30_df,appdf[i]), axis = 1)
    
top_30_df = pd.concat((top_30_df, appdf['United States']), axis = 1)
top_30_df.to_csv('Top_30.csv')

df['mean'] = df.mean(axis=1)



yearly = pd.read_csv('Metro_original.csv')

yearly['RegionName'] = pd.to_datetime(yearly['RegionName'])
yearly.set_index('RegionName', inplace = True)
ys = yearly.groupby([d.year for d in yearly.index]).mean()

filter_ys = pd.DataFrame()

for column in ys:
    na_count = ys[column].isna().sum()
    if na_count < 10:
        filter_ys = pd.concat((filter_ys,ys[column]), axis = 1)


# def diff_function(x):

ys_shift = filter_ys.shift(periods = 1)
ys3 = filter_ys.diff()/ys_shift
ys3 = ys3.loc[2012:]

filter_ys = filter_ys.loc[2012:]

av_column_yr = ys3.mean(axis=0)
av_column_yr = av_column_yr.sort_values(ascending = False)
top30_yr = av_column_yr.index[0:30]
top_30_yr_df = pd.DataFrame()
top_30_orig_df = pd.DataFrame()
for i in top30_yr:
    top_30_yr_df = pd.concat((top_30_yr_df,ys3[i]), axis = 1)
    top_30_orig_df = pd.concat((top_30_orig_df,filter_ys[i]), axis = 1)

top_30_yr_df = pd.concat((top_30_yr_df,ys3['United States']), axis = 1)
top_30_orig_df = pd.concat((top_30_orig_df,filter_ys['United States']), axis = 1)

top_30_yr_df.to_csv('Top_30_yr.csv')
top_30_orig_df.to_csv('Topo_30_orig.csv')

av_column_yr.to_csv('Average Diff.csv')