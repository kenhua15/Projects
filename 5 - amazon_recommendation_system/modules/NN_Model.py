# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 08:27:31 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Lambda

def baseline_model(total_embed_length):
    total = Input((total_embed_length))


    d1 = Dense(2048, activation = 'relu')(total)
    d1 = Dropout(0.2)(d1)
    d1 = BatchNormalization()(d1)
    
    d2 = Dense(1024, activation = 'relu')(d1)
    d2 = Dropout(0.2)(d2)
    d2 = BatchNormalization()(d2)
    
    d3 = Dense(512, activation = 'relu')(d2)
    d3 = Dropout(0.2)(d3)
    d3 = BatchNormalization()(d3)

    d4 = Dense(256, activation = 'relu')(d3)
    d5 = Dense(64, activation = 'relu')(d4)

    d6 = Dense(16, activation = 'relu')(d5)

    out = Dense(1, activation = 'relu')(d6)

    model = Model(inputs=[total], outputs=[out])
    #compile model outside of this function to make it flexible. 
    model.summary()
    return model


def short_model(total_embed_length):
    #Changing it a bit so that the input is the embedding vectors that is generated outside the function. So the input should be two embedding vectors that are not trainable
    total = Input((total_embed_length))


    d4 = Dense(256, activation = 'relu')(total)
    d5 = Dense(64, activation = 'relu')(d4)

    d6 = Dense(16, activation = 'relu')(d5)

    out = Dense(1, activation = 'relu')(d6)

    model = Model(inputs=[total], outputs=[out])
    #compile model outside of this function to make it flexible. 
    model.summary()
    return model

def wide_deep_model(total_embed_length):
    #Changing it a bit so that the input is the embedding vectors that is generated outside the function. So the input should be two embedding vectors that are not trainable
    total = Input((total_embed_length))


    d1 = Dense(8192, activation = 'relu')(total)
    d1 = Dropout(0.2)(d1)
    d1 = BatchNormalization()(d1)
    
    d2 = Dense(4096, activation = 'relu')(d1)
    d2 = Dropout(0.2)(d2)
    d2 = BatchNormalization()(d2)
    
    d3 = Dense(2048, activation = 'relu')(d2)
    d3 = Dropout(0.2)(d3)
    d3 = BatchNormalization()(d3)

    d4 = Dense(1024, activation = 'relu')(d3)
    d4 = Dropout(0.2)(d4)
    d4 = BatchNormalization()(d4)

    d5 = Dense(1024, activation = 'relu')(d4)
    d5 = Dropout(0.2)(d5)
    d5 = BatchNormalization()(d5)
    
    d6 = Dense(512, activation = 'relu')(d5)
    d6 = Dropout(0.2)(d6)
    d6 = BatchNormalization()(d6)

    d7 = Dense(64, activation = 'relu')(d6)

    d8 = Dense(16, activation = 'relu')(d7)

    out = Dense(1, activation = 'relu')(d8)

    model = Model(inputs=[total], outputs=[out])
    #compile model outside of this function to make it flexible. 
    model.summary()
    return model
