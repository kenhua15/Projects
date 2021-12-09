# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:01:30 2021

@author: kenhu
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds


def kNN_model(user_book_matrix):
    from sklearn.neighbors import NearestNeighbors
    
    book_user_matrix = user_book_matrix.T()
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(book_user_matrix)

    return model_knn

def kNN_predict(model_knn, user_book_matrix, bookIDS):
    book_user_matrix = user_book_matrix.T()
    A = model_knn.kneighbors_graph(book_user_matrix)
    A = A.toarray()
    query_index = np.random.choice(book_user_matrix.shape[0])
    distances, indices = model_knn.kneighbors(np.array(book_user_matrix.iloc[query_index,:]).reshape(1,-1), n_neighbors = 6)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            orig_asin = book_user_matrix.index[query_index]
            orig_name = bookIDS.loc[bookIDS['Book-ID'] == orig_asin, 'title']
    
            print('Recommendations for {0}:\n'.format(orig_name))
        else:
            rec_asin = book_user_matrix.index[indices.flatten()[i]]
            rec_name = book_merged.loc[book_merged['Book-ID'] == rec_asin, 'title']
            print('{0}:{1}, with distance of {2}:'.format(i,rec_name, distances.flatten()[i]))




def build_SVD_embeddings(user_book_matrix, embedding_length):
    #Builds SVD embeddings to be used later on
    U, Sigma, VT = svds(user_book_matrix, k = embedding_length)
    user_embed_df = pd.DataFrame(U, index = user_book_matrix.index)
    VT_T = np.transpose(VT)
    book_embed_df = pd.DataFrame(VT_T, index = user_book_matrix.columns)
    
    return user_embed_df, book_embed_df

def SVD_Dot_RSME(user_book_matrix, embedding_length):
    U, Sigma, VT = svds(user_book_matrix, k = embedding_length)
    USigma = np.matmul(U. Sigma)
    pred_matrix = np.matmul(USigma, VT)
    
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(np.array(user_book_matrix), pred_matrix)
    return rmse
    
    
def append_embeddings(df, user_embed_df, book_embed_df):
    #Append embeddings to the full list of rated user-book combinations
    #Returns df with appended embeddings [user] + [book] in a column
    def find_user_book(x):
        #Functions used to 
        user_row = user_embed_df.loc[user_embed_df.index == x[0]]
        book_row = book_embed_df.loc[book_embed_df.index == x[1]]
        return np.concatenate((np.array(user_row),np.array(book_row)), axis = None)
    
    df['Full-Vector'] = df[['User-ID','Book-ID']].apply(find_user_book, axis = 1)

    return df

def create_train_test(df, y_col = 'ReviewRating'):
    #Manipulates the df with Full-Vector to the appropriate format needed for NN modeling.
    #X_train, X_test, y_train, y_test data ready to be fed into NN model
    X_total = df['Full-Vector']

    X_total = np.stack((X_total))
    
    y_total = np.array(df[y_col])
    y_total = np.expand_dims(y_total,axis = 1)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size = 0.2 , random_state = 21)

    return X_train, X_test, y_train, y_test

def create_book_title_list(df):
    #Starting from a df with a list of Book-IDS, we can create a dataframe that is a directory of the bookIDS with the Actual Title of the Book.
    #Could be useful to add category and author too...Think about this
    meta_json_path = 'meta_Books.json'

    size = 100000
    meta = pd.read_json(meta_json_path, lines=True,
                          chunksize=size)
    
    book_ids = pd.DataFrame(df['Book-ID'].unique())
    book_ids = book_ids.rename(columns = {0: 'Book-ID'})
    
    book_list = []
    for chunk_meta in meta:
        book_merged = book_ids.merge(chunk_meta[['asin','title']], how = 'inner' ,left_on ='Book-ID', right_on = 'asin')
        book_list.append(book_merged)
        
    
    book_merged = pd.concat(book_list, ignore_index=True, join='outer', axis=0)
    book_merged2 = book_merged.groupby(by = ['Book-ID','title'], as_index = False).first()

    return book_merged2
    

def get_top_matches(user_embed_df, book_embed_df, combined_final, book_merged, query_index, model, k_matches = 5):
    #dd
    sampled_user_vector = user_embed_df.iloc[[query_index]]
    sampled_user_ID = sampled_user_vector.index[0]
    
    sampled_user_vector = np.array(user_embed_df.iloc[query_index])
    
    prediction_df = pd.DataFrame(index = book_embed_df.index)
    
    prediction_df.reset_index(inplace = True)
    prediction_df['User-ID'] = sampled_user_ID
    
    def build_prediction_embeddings(x):
        book_row = book_embed_df.loc[book_embed_df.index == x]
        return np.concatenate((sampled_user_vector,np.array(book_row)), axis = None)
    
    
    prediction_df['Full-Vector'] = prediction_df['Book-ID'].apply(build_prediction_embeddings)
    
    X_to_train = np.stack(prediction_df['Full-Vector'])
    
    Y_to_train = model.predict(X_to_train)
    
    prediction_df = prediction_df.merge(book_merged[['Book-ID','title']] , how  = 'outer', on = 'Book-ID')
    prediction_df['Trained_Items'] = Y_to_train
    
    already_read = combined_final[combined_final['User-ID'] == sampled_user_ID]
    
    prediction_df_2 = prediction_df.merge(already_read[['Book-ID','ReviewRating']] , how  = 'left', on = 'Book-ID')
    
    top_5_unrated = prediction_df_2.loc[prediction_df_2['ReviewRating'].isna() == True]
    
    top_5_unrated.sort_values(by = ['Trained_Items'], inplace = True, ascending = False)
    
    top_5_unrated = top_5_unrated.iloc[0:k_matches]
    
    
    rated_books = prediction_df_2.loc[prediction_df_2['ReviewRating'].isna() == False]
    rated_books.sort_values(by = ['ReviewRating'], inplace = True, ascending = False)
    top_5_rated = rated_books.iloc[0:k_matches]

    bottom_5_rated = rated_books.iloc[-k_matches:]
    
    number_of_read_books = rated_books.shape
    return top_5_rated, bottom_5_rated, top_5_unrated, number_of_read_books
    
def plot_loss(results):
    #Plot the training loss and the validation loss
    f, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
    t = f.suptitle('CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    
    max_epoch = len(results.history['loss'])+1
    epoch_list = list(range(1,max_epoch))
    ax1.plot(epoch_list, results.history['loss'], label='Train MSE')
    ax1.plot(epoch_list, results.history['val_loss'], label='Validation MSE')
    ax1.set_xticks(np.arange(1, max_epoch, 5))
    ax1.set_ylabel('MSE Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('MSE')
    l1 = ax1.legend(loc="best")

def train_multiple_models(model_list, X_train, y_train, batch_size, epochs):
    results_list = []
    for model_item in model_list:
        print('train')
        results = model_item.fit(X_train,y_train, validation_split = 0.15,batch_size = batch_size, verbose = 1, epochs = epochs)
        results_list.append(results)
    
    return model_list, results_list

def test_multiple_models(model_list, X_test, y_test):
    
    from sklearn.metrics import mean_squared_error
    
    pred_list = []
    mse_list = []
    for model_item in model_list:
        y_pred = model_item.predict(X_test)
        pred_list.append(y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
    
    return mse_list, pred_list
    
    