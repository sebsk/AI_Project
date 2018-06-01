from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def shuffle_train(model, data, text, path, k=10, n_epoch=10):
    min_val_loss = float('inf')
    n_earlystopping = [5]*(k/4)+[4]*(k/4)+[3]*(k/4)+[2]*(k-k/4*3)
    mcp_cnn = ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    history = {'acc':[], 'loss':[], 'val_acc':[], 'val_loss':[]}
    ##### n_epoch*k epochs in total######
    for i in range(k):
        df = data.df.sample(frac=1)
        tweet_g = text[df.index.values]
        his = model.fit(tweet_g, to_categorical(df.label), callbacks=[EarlyStopping(monitor='val_loss', patience=n_earlystopping[i]),
                                mcp_cnn], validation_split=0.3, batch_size=64, epochs=n_epoch)
        for metric in ['acc', 'loss', 'val_acc', 'val_loss']:
            history[metric].extend(his.history[metric])
        if min(his.history['val_loss']) < min_val_loss:
            min_val_loss = min(his.history['val_loss'])
            model.load_weights(path)
        if len(his.history['acc']) > n_earlystopping[i]+1:
            stop = True
            for j in range(1,n_earlystopping[i]+1):
                if his.history['val_loss'][-j] < his.history['val_loss'][-j-1]:
                    stop = False
            if stop: break
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


def plot_history(history, title):
    plt.figure(figsize=(16,6))
    plt.suptitle(title)
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