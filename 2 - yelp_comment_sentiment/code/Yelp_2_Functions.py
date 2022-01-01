# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:45:45 2021

@author: kenhu
"""

import numpy as np
import pandas as pd
import re
from statistics import mean
import nltk
from sklearn.preprocessing import QuantileTransformer


def get_business_dataset(review_threshold):
    #Load restaurant businesses with over a threshold amount of reviews
    business_json_path = 'yelp_academic_dataset_business.json'
    
    df_b = pd.read_json(business_json_path, lines = True)
    df_b = df_b.loc[df_b['review_count'] > review_threshold]
    df_b = df_b.loc[df_b['categories'].str.contains('Restaurants') == True]

    return df_b

def get_review_dataset(review_json_path, businesses):
    
    size = 100000
    review = pd.read_json(review_json_path, lines=True,
                          dtype={'review_id':str,'user_id':str,
                                 'business_id':str,'stars':int,
                                 'date':str,'text':str,'useful':int,
                                 'funny':int,'cool':int},
                          chunksize=size)
    chunk_list = []
    for chunk_review in review:
        # Drop columns that aren't needed
        chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
        # Renaming column name to avoid conflict with business overall star rating
        chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
        # Inner merge with edited business file so only reviews related to the business remain
        chunk_merged = pd.merge(businesses, chunk_review, on='business_id', how='inner')
        # Show feedback on progress
        print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
        chunk_list.append(chunk_merged)
    # After trimming down the review file, concatenate all relevant data back to one dataframe
    df_merged = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)
    return df_merged

def sentiment_analyzer_scores(review):
    
    sentence_list = nltk.tokenize.sent_tokenize(review)
    scores = []
    for sentence in sentence_list:
        for keyword in reg_keyword_list:
            find_list = re.findall(keyword,sentence)
            if len(find_list) > 0:
                count_dict = dict()
                for keyword in reg_keyword_list:
                    counts = sentence.str.count(keyword)
                    if counts > 0:
                        count_dict[keyword] = counts
                    
                sent_df = pd.join(Ratios, pd.DataFrame(), how = 'inner')
                
def topic_modeler(reviews, Ratios):
    #This function creates analyzes each sentence in the reviews, and assigns a topic distribution to them.
    #The sentiment of the sentence will also be analyzed using Flair package
    from flair.models import TextClassifier
    from flair.data import Sentence
    import re
    
    #For checking Flair datatype
    def isfloat(value):
      try:
        float(value)
        return True
      except ValueError:
        return False
    sia = TextClassifier.load('en-sentiment')
    
    #Topic Categorizer to be applied on each review in the dataframe.
    #Return dataframe with each row containing a sentence, sentence topic distribution, and sentence sentiment score
    def topic_categorizer(paragraph):
        #Split paragraph into sentences
        sentence_list = nltk.tokenize.sent_tokenize(paragraph)
        
        #Create Weight list for each topic - we have three topics
        Environment_List = []
        Service_List = []
        Price_List = []

        #Go through each sentence
        sentence_analyzed = []
        sentence_topics = []
        sentence_sentiment = []
        for sentence in sentence_list:
            
            #Create dictionary of all counts of keywords in this sentence
            count_dict = dict()
            for keyword in list(Ratios.index):
                counts = sentence.count(keyword)
                if counts > 0:
                    count_dict[keyword] = counts
            
            #Append this count information to your Term-Ratios Dataframe
            #Calculate the Weighted Ratio of each term
             
            topic_list = []
            if count_dict:
                #Pull the TF-IDF Ratios from the Ratios table. This will weigh each matching word accordingly
                sent_df = Ratios.join(pd.DataFrame.from_dict(count_dict, orient = 'index'), how = 'inner')
                sent_df = sent_df.rename(columns = {0: 'Counts'})
                Ratio_sum = ((1/sent_df['Ratios'])*sent_df['Counts']).sum()
                sent_df['Weighted_Ratio'] = (1/(sent_df['Ratios'])*sent_df['Counts'])/Ratio_sum
                
                #Aggregate the terms into the topics specified
                sent_gb = sent_df.groupby(['Env','Serv','Pr'],as_index = False)['Weighted_Ratio'].sum()
                
                #Append these to the original lists
                if sent_gb['Env'].any() == True:
                    topic_list.append(sent_gb[sent_gb['Env'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    topic_list.append(0)
                if sent_gb['Serv'].any() == True:
                    topic_list.append(sent_gb[sent_gb['Serv'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    topic_list.append(0)
                if sent_gb['Pr'].any() == True:
                    topic_list.append(sent_gb[sent_gb['Pr'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    topic_list.append(0)

                #Calculate the Sentiment of the Sentence
                sent = Sentence(sentence)
                sia.predict(sent)
                score = str(sent.labels[0])
                score = score.replace('(',' ').replace(')',' ')
                number = [float(s) for s in score.split() if isfloat(s) is True]
                
                if "POSITIVE" in score:
                    flair_score =  number[0]
                elif "NEGATIVE" in score:
                    flair_score = -number[0]

            #Append all processed data to the lists.
            if topic_list:
                sentence_topics.append(topic_list)
                sentence_analyzed.append(sentence)
                sentence_sentiment.append(flair_score)
                
        return sentence_analyzed, sentence_topics, sentence_sentiment
    
    review_topic = pd.DataFrame()
    reviews['Sentence'], reviews['Sentence Topics'], reviews['Sentence Sentiment'] = zip(*reviews.text.apply(topic_categorizer))

    return reviews

def topic_modeler_func(reviews, Ratios):
    #This function creates analyzes each sentence in the reviews, and assigns a topic distribution to them.
    
    #Topic Categorizer to be applied on each review in the dataframe.
    #Return dataframe with each row containing a sentence, sentence topic distribution
    def topic_categorizer(paragraph):
        #Split paragraph into sentences
        sentence_list = nltk.tokenize.sent_tokenize(paragraph)
        
        #Create Weight list for each topic - we have three topics
        Environment_List = []
        Service_List = []
        Price_List = []

        #Go through each sentence
        sentence_analyzed = []
        sentence_topics = []
        for sentence in sentence_list:
            
            #Create dictionary of all counts of keywords in this sentence
            count_dict = dict()
            for keyword in list(Ratios.index):
                counts = sentence.count(keyword)
                if counts > 0:
                    count_dict[keyword] = counts
            
            #Append this count information to your Term-Ratios Dataframe
            #Calculate the Weighted Ratio of each term
             
            topic_list = []
            if count_dict:
                #Pull the TF-IDF Ratios from the Ratios table. This will weigh each matching word accordingly
                sent_df = Ratios.join(pd.DataFrame.from_dict(count_dict, orient = 'index'), how = 'inner')
                sent_df = sent_df.rename(columns = {0: 'Counts'})
                Ratio_sum = ((1/sent_df['Ratios'])*sent_df['Counts']).sum()
                sent_df['Weighted_Ratio'] = (1/(sent_df['Ratios'])*sent_df['Counts'])/Ratio_sum
                
                #Aggregate the terms into the topics specified
                sent_gb = sent_df.groupby(['Env','Serv','Pr'],as_index = False)['Weighted_Ratio'].sum()
                
                #Append these to the original lists
                if sent_gb['Env'].any() == True:
                    topic_list.append(sent_gb[sent_gb['Env'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    topic_list.append(0)
                if sent_gb['Serv'].any() == True:
                    topic_list.append(sent_gb[sent_gb['Serv'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    topic_list.append(0)
                if sent_gb['Pr'].any() == True:
                    topic_list.append(sent_gb[sent_gb['Pr'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    topic_list.append(0)

            #Append all processed data to the lists.
            if topic_list:
                sentence_topics.append(topic_list)
                sentence_analyzed.append(sentence)
                
        return sentence_analyzed, sentence_topics
    
    review_topic = pd.DataFrame()
    reviews['Sentence'], reviews['Sentence Topics'] = zip(*reviews.text.apply(topic_categorizer))

    return reviews

def process_split_df(reviews):
    review_names = reviews[['business_id','user_id','name','review_stars']]

    reviews_split = reviews[['business_id','user_id', 'Sentence', 'Sentence Topics', 'Sentence Sentiment']]

    reviews_split = reviews_split.set_index(['business_id','user_id']).apply(pd.Series.explode).reset_index()
    reviews_split = reviews_split[~reviews_split['Sentence Topics'].isnull()]

    reviews_processed = pd.merge(review_names, reviews_split, on = ['business_id','user_id'], how = 'inner')

    return reviews_processed

def quantile_transform(df):
    
    from sklearn.preprocessing import QuantileTransformer
    reviews_y = df[['Sentence Sentiment']]
    
    qt = QuantileTransformer()
    qt2 = QuantileTransformer()
    X_1 = reviews_y.loc[reviews_y['Sentence Sentiment'] >= 0]
    X_2 = reviews_y.loc[reviews_y['Sentence Sentiment'] < 0]
    
    X_1['flair_quantile_custom'] = qt.fit_transform(X_1[['Sentence Sentiment']])/2 + 0.5
    X_2['flair_quantile_custom'] = qt2.fit_transform(X_2[['Sentence Sentiment']])/2
    
    X = X_1.append(X_2)
    X.sort_index(inplace = True)
    
    return X['flair_quantile_custom']


def data_aggregator(reviews_split):
    
    #First split the Sentence Topics column into columns for your topics. In my case: Env, Serv, Pr
    def split_topics(x):
        return x[0],x[1],x[2]

    reviews_split['Env'], reviews_split['Serv'], reviews_split['Pr'] =  zip(*reviews_split['Sentence Topics'].apply(split_topics))

    #Any Topic Percentages with 0s should be replaced with nan
    def set_nan(x):
        if x == 0:
            return np.nan
        else:
            return x
    
    
    
    reviews_split['Env'] = reviews_split['Env'].apply(set_nan)
    reviews_split['Serv'] = reviews_split['Serv'].apply(set_nan)
    reviews_split['Pr'] = reviews_split['Pr'].apply(set_nan)

    #Now, for each sentence, we weigh the score by normalizing it against the proportion across the entire restaurant. We will sum this up later.
    #So this is a way to proportionalize the sentiment score based on topic percentages for all sentences in a review.
    def weight_sentiment(df, x,y):
        return (df[x]*df[y])/(df[x].sum())

    grouped_env = pd.DataFrame(reviews_split.groupby(['business_id']).apply(weight_sentiment, x = 'Env', y = 'Sentence Sentiment')) 
    grouped_env = grouped_env.rename(columns = {0: 'Env'})
    grouped_serv = pd.DataFrame(reviews_split.groupby(['business_id']).apply(weight_sentiment, x = 'Serv', y = 'Sentence Sentiment'))
    grouped_serv = grouped_serv.rename(columns = {0: 'Serv'})
    grouped_price = pd.DataFrame(reviews_split.groupby(['business_id']).apply(weight_sentiment, x = 'Pr', y = 'Sentence Sentiment'))  
    grouped_price = grouped_price.rename(columns = {0: 'Pr'})

    #Sum up all weighted sentences for each review, to obtain the final Sentiment Score for each topic for each restaurant
    def check_nan(df, x):
        if df[x].isnull().all() == True:
            return np.nan
        else:
            return df[x].sum()
    res_env = grouped_env.groupby(['business_id']).apply(check_nan, x = 'Env')
    res_env = res_env.reset_index().rename(columns = {0: 'Env'})
    res_serv = grouped_serv.groupby(['business_id']).apply(check_nan, x = 'Serv')
    res_serv = res_serv.reset_index().rename(columns = {0 : 'Serv'})
    res_price = grouped_price.groupby(['business_id']).apply(check_nan, x = 'Pr')
    res_price = res_price.reset_index().rename(columns = {0 : 'Pr'})

    #Now get ratings for each restaurant, and let's combine the columns to get the final dataframe
    res_rating = reviews_split.groupby(['business_id','name'], as_index = False)['review_stars'].mean()

    res_rating = res_rating.merge(res_env, left_on = 'business_id', right_on = 'business_id')
    res_rating = res_rating.merge(res_serv, left_on = 'business_id', right_on = 'business_id')
    res_rating = res_rating.merge(res_price, left_on = 'business_id', right_on = 'business_id')
    
    return res_rating


def topic_modeler_V1(reviews, Ratios):    
    def topic_categorizer(paragraph):
        #Split paragraph into sentences
        sentence_list = nltk.tokenize.sent_tokenize(paragraph)
        
        #Create Weight list for each topic
        Environment_List = []
        Service_List = []
        Price_List = []

        #Go through each sentence
        for sentence in sentence_list:
            
            #Create dictionary of all counts of keywords in this sentence
            count_dict = dict()
            for keyword in list(Ratios.index):
                counts = sentence.count(keyword)
                if counts > 0:
                    count_dict[keyword] = counts
            
            #Append this count information to your Term-Ratios Dataframe
            #Calculate the Weighted Ratio of each term
            if count_dict:
                sent_df = Ratios.join(pd.DataFrame.from_dict(count_dict, orient = 'index'), how = 'inner')
                sent_df = sent_df.rename(columns = {0: 'Counts'})
                Ratio_sum = ((1/sent_df['Ratios'])*sent_df['Counts']).sum()
                sent_df['Weighted_Ratio'] = (1/(sent_df['Ratios'])*sent_df['Counts'])/Ratio_sum
                
                #Aggregate the terms into the topics specified
                sent_gb = sent_df.groupby(['Env','Serv','Pr'],as_index = False)['Weighted_Ratio'].sum()
                
                #Append these to the original lists
                if sent_gb['Env'].any() == True:
                    Environment_List.append(sent_gb[sent_gb['Env'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    Environment_List.append(0)
                if sent_gb['Serv'].any() == True:
                    Service_List.append(sent_gb[sent_gb['Serv'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    Service_List.append(0)
                if sent_gb['Pr'].any() == True:
                    Price_List.append(sent_gb[sent_gb['Pr'] == True].reset_index().iloc[0]['Weighted_Ratio'])
                else:
                    Price_List.append(0)
            
        
        
        if not Environment_List:
            return [0]
        else:
            #Take average of these lists, which are the average topic distributions of sentences containing keywords in this review

            Environment_Avg = mean(Environment_List)   
            Service_Avg = mean(Service_List) 
            Price_Avg = mean(Price_List)
            
            topic_distribution = [Environment_Avg, Service_Avg, Price_Avg]


            return topic_distribution  
            

    review_topic = reviews.text.apply(topic_categorizer)

    return review_topic            
            
            