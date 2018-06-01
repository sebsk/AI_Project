from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout


def create_cnn_model(conv=[32, 16], window=[3,3], pool=3, dropout=0.3, penalty=0.01, input_shape=25, batch_normalize=False):
    model = Sequential()
    model.add(Conv1D(conv[0], window[0], activation='relu', kernel_initializer = 'he_normal',
                   kernel_regularizer=regularizers.l2(penalty), input_shape=(None,input_shape)))
    model.add(MaxPooling1D(pool))
    model.add(Conv1D(conv[1], window[1], activation='relu', kernel_initializer = 'he_normal',
                   kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalMaxPooling1D())
    if batch_normalize:
        model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model