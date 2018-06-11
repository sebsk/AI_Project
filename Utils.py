from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K

sns.set()


def shuffle_train(model, data, text, path, k=10, n_epoch=10, val_size=0.2, batch_size=64, earlystop=None):
    min_val_loss = float('inf')
    if isinstance(earlystop, int):
        n_earlystopping = [earlystop]*k
    elif isinstance(earlystop, list) and len(earlystop)==k:
        n_earlystopping = earlystop
    else:
        print 'earlystop is not specified or the input is invalid(has either to be an int or a list with length equal to k), use default setting'
        n_earlystopping = [7]*(k/4)+[6]*(k/4)+[5]*(k/4)+[4]*(k-k/4*3)
    mcp = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, mode='auto', cooldown=0, min_lr=0.0001)
    history = {'acc':[], 'loss':[], 'val_acc':[], 'val_loss':[]}
    ##### n_epoch*k epochs in total######
    CLR = K.get_value(model.optimizer.lr)
    for i in range(k):
        K.set_value(model.optimizer.lr, CLR)
        df = data.df.sample(frac=1)
        tweet_g = text[df.index.values]
        his = model.fit(tweet_g, to_categorical(df.label), callbacks=[EarlyStopping(monitor='val_loss', patience=n_earlystopping[i], min_delta=0.001),
            mcp, rlr], validation_split=val_size, batch_size=batch_size, epochs=n_epoch)
        for metric in ['acc', 'loss', 'val_acc', 'val_loss']:
            history[metric].extend(his.history[metric])
        if min(his.history['val_loss']) < min_val_loss:
            min_val_loss = min(his.history['val_loss'])
        if len(his.history['acc']) < n_epoch:
            break
        CLR = K.get_value(model.optimizer.lr)
        model.load_weights(path)
    model.save(path)
    return history


def metric_drawing(data_list, labels, title, Model=None, path=None): # data_list is dict with tweets as items. labels is dict with labels as items
    if Model is not None and path is not None:
        print 'Choose either Model or path but not both!'
        return
    if Model is None:
        model = load_model(path)
    else:
        model = Model
    y_pred = {}
    precision = {}
    recall = {}
    f1 = {}
    data_num = len(data_list.keys())
    for i in data_list.keys():
        y_pred_data = []
        y_p = model.predict(data_list[i])
        for y in y_p:
            if y[0] > y[1]:
                y_pred_data.append(0)
            else:
                y_pred_data.append(1)
        y_pred.update({i:y_pred_data})
        precision.update({i:metrics.precision_score(labels[i], y_pred_data)})
        recall.update({i:metrics.recall_score(labels[i], y_pred_data)})
        f1.update({i:metrics.f1_score(labels[i], y_pred_data)})

    plt.bar(np.arange(1, data_num+1)-0.1, precision.items(), width=0.1, label='precision')
    plt.bar(np.arange(1, data_num+1), recall.items(), width=0.1, label='recall')
    plt.bar(np.arange(1, data_num+1)+0.1, f1.items(), width=0.1, label='f1-score')
    plt.xticks(np.arange(1, data_num+1), data_list.keys())
    plt.ylim([0,1])
    plt.legend(loc='best')
    plt.xlabel('disaster')
    plt.ylabel('score')
    plt.title(title)
    plt.show()
    return precision, recall, f1


def plot_history(history, title, dictionary=False):
    plt.figure(figsize=(16,6))
    plt.suptitle(title)
    if dictionary:
        plt.subplot(121)
        n_epoch = len(history['acc'])
        plt.plot(range(1, n_epoch+1), history['loss'], label='loss')
        plt.plot(range(1, n_epoch+1), history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(range(1, n_epoch+1), history['acc'], label='acc')
        plt.plot(range(1, n_epoch+1), history['val_acc'], label='val_acc')
    else:
        plt.subplot(121)
        n_epoch = len(history.history['acc'])
        plt.plot(range(1, n_epoch+1), history.history['loss'], label='loss')
        plt.plot(range(1, n_epoch+1), history.history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(range(1, n_epoch+1), history.history['acc'], label='acc')
        plt.plot(range(1, n_epoch+1), history.history['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()