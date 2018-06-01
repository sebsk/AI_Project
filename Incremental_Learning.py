from keras.utils import to_categorical
import numpy as np


def incremental_learning_prob(model, tweets, step=10, threshold=0.75):
    subset_len = tweets.shape[0]/step
    marks = [subset_len*i for i in range(step)]
    marks.append(tweets.shape[0])
    for i in range(step):
        text = tweets[marks[i]: marks[i+1]]
        y_pred = model.predict(x=text)
        x_confident = np.asarray([text[j] for j in range(len(y_pred)) if y_pred[j][0] >= threshold or y_pred[j][1] >= threshold])
        y_confident = [y_pred[j] for j in range(len(y_pred)) if y_pred[j][0] >= threshold or y_pred[j][1] >= threshold]
        label = to_categorical(map(lambda x: 0 if x[0]>=threshold else 1, y_confident))
        weight = np.array([(max(i)-0.5)/0.5 for i in y_confident])
        model.train_on_batch(x_confident, label, sample_weight=weight)


def incremental_learning_det(model, tweets, step=10, confidence=0.95):
    subset_len = tweets.shape[0]/step
    marks = [subset_len*i for i in range(step)]
    marks.append(tweets.shape[0])
    for i in range(step):
        text = tweets[marks[i]: marks[i+1]]
        y_pred = model.predict(x=text)
        x_confident = np.asarray([text[j] for j in range(len(y_pred)) if y_pred[j][0] >= confidence or y_pred[j][1] >= confidence])
        y_confident = [y_pred[j] for j in range(len(y_pred)) if y_pred[j][0] >= confidence or y_pred[j][1] >= confidence]
        label = to_categorical(map(lambda x: 0 if x[0]>=confidence else 1, y_confident))
        model.train_on_batch(x_confident, label)