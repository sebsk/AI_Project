from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import Dropout


def create_lstm_model(bidir=True, n_lstm=32, dropout=0.3, re_dropout=0.3, penalty=0.01, input_shape=25, initial_lr=0.001, dense=False, n_dense=None):
    model = Sequential()
    if bidir:
        model.add(Bidirectional(LSTM(n_lstm, kernel_initializer='he_normal', recurrent_dropout=re_dropout, dropout=dropout,
                                     kernel_regularizer=regularizers.l2(penalty)),input_shape=(None,input_shape)))
    else:
        model.add(LSTM(n_lstm, kernel_initializer='he_normal', recurrent_dropout=re_dropout, dropout=dropout,
                                     kernel_regularizer=regularizers.l2(penalty),input_shape=(None,input_shape)))
    if dense:
        if n_dense is None:
            n_dense = 64
        model.add(Dense(n_dense, activation='relu',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(penalty)))
        model.add(Dropout(0.3))
    optimizer = Adam(lr=initial_lr)
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model