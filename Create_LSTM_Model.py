from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras import regularizers


def create_lstm_model(bidir=True, n_lstm=32, dropout=0.3, re_dropout=0.3, penalty=0.01, input_shape=25):
    model = Sequential()
    if bidir:
        model.add(Bidirectional(LSTM(n_lstm, kernel_initializer='he_normal', recurrent_dropout=re_dropout, dropout=dropout,
                                     kernel_regularizer=regularizers.l2(penalty)),input_shape=(None,input_shape)))
    else:
        model.add(LSTM(n_lstm, kernel_initializer='he_normal', recurrent_dropout=re_dropout, dropout=dropout,
                                     kernel_regularizer=regularizers.l2(penalty),input_shape=(None,input_shape)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model