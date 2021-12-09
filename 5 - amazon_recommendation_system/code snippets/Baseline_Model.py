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


model = baseline_model(64*2)
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
