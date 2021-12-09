# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:31:18 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filter_comment_lengths(reviews, minimum, maximum):
    #Filters reviews based on word length, between a minimum and maximum
    #Returns the df with word length as an additional column, and filtered rows of comments between min and max
    reviews['reviewText'] = reviews['reviewText'].astype('str')
    reviews['reviewLen'] = reviews['reviewText'].str.split().map(lambda x: len(x))
    reviews = reviews.loc[(reviews['reviewLen'] > minimum) & (reviews['reviewLen'] < maximum)]
    
    return reviews

def vader_sentiment(reviews):
    #Apply Vader Sentiment Analysis on your df
    #Appends all the different sentiments to your df
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    
    def sentiment_analyzer_scores(sentence):
        score = analyser.polarity_scores(sentence)
        return score

    reviews['text_sentiment'] = reviews['reviewText'].apply(sentiment_analyzer_scores)

    reviews['compound'] = [d.get('compound') for d in reviews['text_sentiment']]
    reviews['neg'] = [d.get('neg') for d in reviews['text_sentiment']]
    reviews['neu'] = [d.get('neu') for d in reviews['text_sentiment']]
    reviews['pos'] = [d.get('pos') for d in reviews['text_sentiment']]

    return reviews

def textblob_sentiment(reviews):
    #Uses Textblob to analyze polarity and subjectivity in your comments
    from textblob import TextBlob

    pol = lambda x: TextBlob(x).sentiment.polarity
    sub = lambda x: TextBlob(x).sentiment.subjectivity
    
    reviews['polarity'] = reviews['reviewText'].apply(pol)
    reviews['subjectivity'] = reviews['reviewText'].apply(sub)

    return reviews

def flair_sentiment(df):
    #Uses flair package to analyze text sentiment classification. I also add a quantiletranformation to the metric, which is more suitable for learning
    #Returnd DataFrame with raw sentiment score and quantile transformed score, obtained from Flair models
    from flair.models import TextClassifier
    from flair.data import Sentence
    import re
    
    def isfloat(value):
      try:
        float(value)
        return True
      except ValueError:
        return False
    
    sia = TextClassifier.load('en-sentiment')
    def flair_prediction(x):
        sentence = Sentence(x)
        sia.predict(sentence)
        score = str(sentence.labels[0])
        score = score.replace('(',' ').replace(')',' ')
        number = [float(s) for s in score.split() if isfloat(s) is True]
        
        if "POSITIVE" in score:
            return number[0]
        elif "NEGATIVE" in score:
            return -number[0]
        
    df["flair_sentiment"] = df["reviewText"].apply(flair_prediction)

    
    return df

def quantile_transform(df):
    
    from sklearn.preprocessing import QuantileTransformer
    reviews_y = df[['flair_sentiment','flair_quantile']]
    
    qt = QuantileTransformer()
    qt2 = QuantileTransformer()
    X_1 = reviews_y.loc[reviews_y['flair_sentiment'] >= 0]
    X_2 = reviews_y.loc[reviews_y['flair_sentiment'] < 0]
    
    X_1['flair_quantile_custom'] = qt.fit_transform(X_1[['flair_sentiment']])/2 + 0.5
    X_2['flair_quantile_custom'] = qt2.fit_transform(X_2[['flair_sentiment']])/2
    
    X = X_1.append(X_2)
    X.sort_index(inplace = True)
    
    return X['flair_quantile_custom']

def evaluate_sentiments(df):
    #Print randomly sampled row to evaluate how the flair is matching up to the comment
    samp_text = df.sample(1)
    pd.set_option('display.max_colwidth', None)
    
    print(samp_text)

def make_scattertext(df, ratingcol):
    
    import scattertext as st
    #Make ScatterText Plot
    df['Liked'] = df[ratingcol].apply(lambda x: x > 0.5).map({True: 'Liked', False: 'Not Liked'})

    corpus = st.CorpusFromPandas(
        df,
        category_col = 'Liked',
        text_col = 'reviewText',
        nlp=st.whitespace_nlp_with_sentences
    ).build()
        
    html = st.produce_scattertext_explorer(
        corpus,
        category="Liked",
        category_name='Liked',
        not_category_name='Not Liked',
        minimum_term_frequency=10,
        pmi_threshold_coefficient=5,
        width_in_pixels=1000,
        metadata=df['Book-ID']
        )

    open('reviews_scatter_flair2.html', 'wb').write(html.encode('utf-8'));
        