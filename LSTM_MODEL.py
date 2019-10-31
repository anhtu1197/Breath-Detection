import numpy as np
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Flatten
from keras.layers.merge import add
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model

class LSTM_MODEL(object):
    @staticmethod
    def build_simple_lstm(data_input_shape, classes, learning_rate):
        model = Sequential()
        model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=data_input_shape))
        model.add(LSTM(units=128,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=classes, activation="softmax"))
        # Keras optimizer defaults:
        # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
        # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
        # SGD    : lr=0.01,  momentum=0.,                             decay=0.
        opt = Adam(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
    
    @staticmethod
    def build_bilstm(data_input_shape, classes, learning_rate):
        model = Sequential()
        model.add(Bidirectional(LSTM(128), input_shape=data_input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        opt = Adam(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()

        return model

    @staticmethod
    def build_residual_bilstm(data_input_shape, classes, learning_rate):
        inp = Input(shape=data_input_shape)
        z1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
        z2 = Bidirectional(LSTM(units=128, return_sequences=True))(z1)
        z3 = add([z1, z2])  # residual connection
        z4 = Bidirectional(LSTM(units=128, return_sequences=True))(z3)
        z5 = Bidirectional(LSTM(units=128, return_sequences=False))(z4)
        z6 = add([z4, z5])  # residual connection    
        z61 = Flatten()(z6)        
        z7 = Dense(256, activation='relu')(z61)
        z8 = Dropout(0.5)(z7)
        out = Dense(classes, activation='softmax')(z8)
        model = Model(inputs=[inp], outputs=out)
        opt = Adam(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()
        return model

