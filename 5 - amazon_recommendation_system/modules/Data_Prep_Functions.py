# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 23:23:35 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def user_book_threshold(sample_list, user_threshold, book_threshold):
    #Function that takes sample_list and only selects users and books above a threshold. 
    #Returns cleaned dataframe of USER-ID, BOOK-ID, and Scaled RATING for items above the threshold count
    book_count = (sample_list.
         groupby(by = ['asin'])['overall'].
         count().
         reset_index()
        )
    
    
    book_count = book_count.query('overall >= @book_threshold')
    
    user_rating = pd.merge(book_count, sample_list, left_on='asin', right_on='asin', how='left')
    
    user_count = (sample_list.
         groupby(by = ['reviewerID'])['overall'].
         count().
         reset_index()
        )
    
    user_count = user_count.query('overall >= @user_threshold')
    
    combined = user_rating.merge(user_count, left_on = 'reviewerID', right_on = 'reviewerID', how = 'inner')
    combined_gb = combined.groupby(by = ['reviewerID','asin'],as_index=False).mean()

    combined_final = combined_gb.drop(columns = ['overall_x','overall'])
    combined_final = combined_final.rename(columns = {'reviewerID':'User-ID', 'asin': 'Book-ID', 'overall_y':'Rating'})
    # scaler = MinMaxScaler()
    # combined_final['Rating'] = combined_final['Rating'].values.astype(float)
    # rating_scaled = pd.DataFrame(scaler.fit_transform(combined_final['Rating'].values.reshape(-1,1)))
    # combined_final['Rating'] = rating_scaled

    return combined_final


def user_book_threshold_reviews(sample_list, user_threshold, book_threshold):
    #Function that takes sample_list and only selects users and books above a threshold. 
    #Returns cleaned dataframe of USER-ID, BOOK-ID, and Scaled RATING for items above the threshold count
    book_count = (sample_list.
         groupby(by = ['asin'])['overall'].
         count().
         reset_index()
        )
    
    
    book_count = book_count.query('overall >= @book_threshold')
    
    user_rating = pd.merge(sample_list, book_count, left_on='asin', right_on='asin', how='inner')
    
    user_count = (sample_list.
         groupby(by = ['reviewerID'])['overall'].
         count().
         reset_index()
        )
    
    user_count = user_count.query('overall >= @user_threshold')
    
    combined = user_rating.merge(user_count, left_on = 'reviewerID', right_on = 'reviewerID', how = 'inner')
    combined_gb = combined.groupby(by = ['reviewerID','asin', 'reviewText'],as_index=False).mean()

    combined_final = combined_gb.drop(columns = ['overall_x','overall'])
    combined_final = combined_final.rename(columns = {'reviewerID':'User-ID', 'asin': 'Book-ID', 'overall_y':'Rating'})
    # scaler = MinMaxScaler()
    # combined_final['Rating'] = combined_final['Rating'].values.astype(float)
    # rating_scaled = pd.DataFrame(scaler.fit_transform(combined_final['Rating'].values.reshape(-1,1)))
    # combined_final['Rating'] = rating_scaled

    return combined_final
