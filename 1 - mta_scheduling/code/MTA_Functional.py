#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 21:32:21 2021

@author: kenhua
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

def loaddata(img_file): 
    """This function serves to load the csv data into a dataframe. It will also append a date_time column"""
    main = pd.read_csv(img_file)      
    main['DATE_TIME'] = pd.to_datetime(main.DATE + ' ' + main.TIME)
    return main

def daily_data(main, line = '7', season = 2, year = '2021'): 
    """This function serves to clean/aggregate the dataset into daily turnstile data"""
    
    """This part serves to calculate the daily entries based on these cumulative entry and exit data"""
    daily = main.groupby(['C/A','UNIT','SCP','STATION','LINENAME','DATE'], as_index = False)[['ENTRIES','EXITS']].max()
    daily['DAILY_ENTRIES'] = daily.groupby(['C/A'  ,'UNIT','SCP','STATION'])['ENTRIES'].diff().fillna(0)
    daily = daily.loc[daily['DAILY_ENTRIES'] > 0]
    daily = daily.loc[daily['DAILY_ENTRIES'] < 10000]
    daily['DAILY_EXITS'] = daily.groupby(['C/A'  ,'UNIT','SCP','STATION'])['EXITS'].diff().fillna(0)
    daily = daily.loc[daily['DAILY_EXITS'] > 0]
    daily = daily.loc[daily['DAILY_EXITS'] < 10000]
    
    
    """This part selects the rows corresponding to the Line specified in the function, it also selects the rows corresponding to """

    daily_line_7 = daily.loc[daily['LINENAME'].str.contains(line)]
    
    daily_line_group = daily_line_7.groupby('DATE',as_index = False)[['DAILY_ENTRIES','DAILY_EXITS']].sum()
    daily_line_group['TOTAL_ENTRIES_EXITS'] = daily_line_group['DAILY_ENTRIES'] + daily_line_group['DAILY_EXITS']
    daily_line_group['3_DAY_MOVING_AVERAGE'] = daily_line_group['TOTAL_ENTRIES_EXITS'].rolling(window=3).mean()


    """selects the rows corresponding to season, default is just spring"""

    daily_line_group['SEASON'] = pd.to_datetime(daily_line_group.DATE)
    daily_line_group['SEASON'] = daily_line_group['SEASON'].dt.month%12 // 3 + 1
    daily_line_group_spring = daily_line_group.loc[daily_line_group['SEASON'] == season]    
    daily_line_group_spring['YEAR'] = year
    

    return daily_line_group_spring

def hourly_week_data(main, line = '7', year = '2021'):
    """This function serves to clean/aggregate the dataset into weekly-hourly turnstile data"""

    """This part serves to calculate the hourly (or 4 hour) entries based on the cumulative entry and exit data"""
    hourly = main.groupby(['C/A','UNIT','SCP','STATION','LINENAME','DATE','TIME', 'DATE_TIME'], as_index = False)[['ENTRIES','EXITS']].max()
    hourly['HOURLY_ENTRIES'] = hourly.groupby(['C/A'  ,'UNIT','SCP','STATION'])['ENTRIES'].diff().fillna(0)
    hourly = hourly.loc[hourly['HOURLY_ENTRIES'] > 0]
    hourly = hourly.loc[hourly['HOURLY_ENTRIES'] < 10000]
    hourly['HOURLY_EXITS'] = hourly.groupby(['C/A'  ,'UNIT','SCP','STATION'])['EXITS'].diff().fillna(0)
    hourly = hourly.loc[hourly['HOURLY_EXITS'] > 0]
    hourly = hourly.loc[hourly['HOURLY_EXITS'] < 10000]
    
    
    #Selecting for line
    hourly_line_7 = hourly.loc[hourly['LINENAME'].str.contains(line)]
    
    
    #There are some Time values that are off from regular, periodic intervals. This code groups them
    #into regular 4 hour intervals as defined by the dictionary shown.
    #After the Times are binned into intervals, there are only 6 time intervals
    #A day of the week is also added to it via dt.dayofweek
    #Then finally it can be grouped into day of week + time combination, which is 7 * 6 = 42 rows.
    
    hourly_group_7 = hourly_line_7
    hourly_group_7.set_index('DATE_TIME',inplace= True)
    hourly_group_7 = hourly_group_7.groupby(['C/A','UNIT','SCP','STATION','DATE',pd.Grouper(freq='240Min', base=120, label='right')]).sum()
    hourly_group_7 = hourly_group_7.reset_index()
    hourly_group_7['TIME'] = hourly_group_7.DATE_TIME.dt.time
    hourly_group_7['DAY_OF_WEEK'] = hourly_group_7['DATE_TIME'].dt.dayofweek
    hourly_group_7['DAY_OF_WEEK_NAME'] = hourly_group_7['DATE_TIME'].dt.day_name()
    
    duration = dict({2:'20:00-00:00',6:'00:00-04:00',10:'04:00-08:00',14:'08:00-12:00',18:'12:00-16:00',22:'16:00-20:00'})
    times = pd.DatetimeIndex(hourly_group_7['DATE_TIME'])
    hourly_group_7['HOUR'] = times.hour
    hourly_group_7['TIME_RANGE'] = hourly_group_7['HOUR'].map(duration)
    
    
    hourly_7 = hourly_group_7.groupby(['HOUR','TIME_RANGE'], as_index = False)[['HOURLY_ENTRIES','HOURLY_EXITS']].sum()
    hourly_week_group_7 = hourly_group_7.groupby(['DAY_OF_WEEK','DAY_OF_WEEK_NAME','TIME_RANGE','HOUR'], as_index = False)[['HOURLY_ENTRIES','HOURLY_EXITS']].sum()
    hourly_week_group_7['TOTAL_ENTRIES_EXITS'] = hourly_week_group_7['HOURLY_ENTRIES'] + hourly_week_group_7['HOURLY_EXITS']
    hourly_week_group_7['YEAR'] = year
    
    return hourly_week_group_7

def plot_daily(daily, year):
    """Plots one year of daily turnstile data on a single plot"""

    plt.figure(figsize=(10,5))
    plt.plot(daily['DATE'],daily['TOTAL_ENTRIES_EXITS'])
    
    plt.xticks(daily['DATE'][::7],  rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Total Entries and Exits')
    plt.title('Line 7 total ridership (Entries + Exits) in Spring 2019')
    plt.tight_layout()
    plt.savefig(year + '_DAILY.png')

    plt.clf()
    
    
def plot_daily_combined(daily):
    """Plots multiple years of daily turnstile data on a single plot"""
    unique_vals = daily['Year'].unique()
    plt.figure(figsize=(10,5))
    for i in unique_vals:
        data = daily.loc[daily['Year'] == i]
        plt.plot('DATE', 'TOTAL_ENTRIES_EXITS', data = data)
    
    plt.xticks(daily['DATE'][::7],  rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Total Entries and Exits')
    plt.title('Line 7 total ridership (Entries + Exits) in Spring 2019')
    plt.tight_layout()
    plt.savefig('Total_DAILY.png')
    plt.clf() 

def plot_daily_stacked(daily):
    """Plots multiple years of daily turnstile data on a stacked plot"""

    unique_vals = daily['YEAR'].unique()
    unique_vals.sort()
    
    fig, axs = plt.subplots(len(unique_vals),figsize = (15,15))
    fig.suptitle('3 Day moving average of Entries and Exits on Line 7 from 2017-2019', fontsize = 30)
    plt.ylabel('3-Day Moving Average of Entries and Exits', fontsize = 20)
    
    for index, i in enumerate(unique_vals):
        data = daily.loc[daily['YEAR'] == i]
        axs[index].plot('DATE', '3_DAY_MOVING_AVERAGE', data = data)
        axs[index].set_xticks(data['DATE'][::7])
        axs[index].tick_params(labelrotation=45)
        axs[index].set_title(str(i), fontsize = 20)
    fig.tight_layout()
    
    plt.xlabel('Date', fontsize = 20)
    # fig.title('Line 7 total ridership (Entries + Exits) in Spring 2019')
    # fig.tight_layout()

    fig.savefig('Total_DAILY_STACKED2.png')
    fig.show()
    return fig

def plot_hourly(hourly, year):
    """Plots weekly-hourly on a stacked plot"""

    plt.figure(figsize=(8,8))

    colors = list(hourly['TIME_RANGE'].unique())
    for i in range(0 , len(colors)):
        data = hourly.loc[hourly['TIME_RANGE'] == colors[i]]
        plt.scatter('DAY_OF_WEEK_NAME', 'TOTAL_ENTRIES_EXITS', data=data, label=colors[i])
    plt.legend()
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Entries and Exits')
    
    plt.title('Average ' + str(year) + ' total ridership per Time range per day from 2017-2019')
    plt.savefig(year + '_HOURLY.png')
    
    plt.show()
