import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class dataset():
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        col_dict = {}
        for old_feature in self.df.columns.values:
            col_dict.update({old_feature: old_feature.replace(' ', '')})
        self.df.rename(columns=col_dict, inplace=True)
        self.df = self.df[self.df.Informativeness != 'Not applicable']  # dataframe
        self.le = LabelEncoder()  # labelencoder
        self.le.fit(self.df.Informativeness)
        tokens = []
        for text in self.df.TweetText:
            for word in text_to_word_sequence(text):
                tokens.append(word)
        self.total_words = len(set(tokens))  # total_words
        self.tokenizer = None

    def generate_tokenizer(self, n_word=None):
        if n_word is None:
            n_word = self.total_words
        self.tokenizer = Tokenizer(num_words=n_word)  # tokenizer
        self.tokenizer.fit_on_texts(self.df.TweetText)
        self.vocab_size = n_word  # vocab_size

    def embedding(self, external_tokenizer=None):  # return embedding texts
        if external_tokenizer is None and self.tokenizer is None:
            return "please generate tokenizer first!"
        if external_tokenizer is None:
            encoded_tweets = self.tokenizer.texts_to_sequences(self.df.TweetText)
            padded_tweets = pad_sequences(encoded_tweets, maxlen=140, padding='post')
            return padded_tweets
        encoded_tweets = external_tokenizer.texts_to_sequences(self.df.TweetText)
        padded_tweets = pad_sequences(encoded_tweets, maxlen=140, padding='post')
        return padded_tweets

    def bow(self, m='binary', external_tokenizer=None):  # return vectorized texts, m can be binary, count, tfdif, freq
        if external_tokenizer is None and self.tokenizer is None:
            return "please generate tokenizer first!"
        if external_tokenizer is None:
            encoded_tweets = self.tokenizer.texts_to_matrix(self.df.TweetText, mode=m)
            return encoded_tweets
        encoded_tweets = external_tokenizer.texts_to_matrix(self.df.TweetText, mode=m)
        return encoded_tweets

    def label(self, external_le=None):  # return encoded label
        if external_le is None:
            label = self.le.transform(self.df.Informativeness)
            label_encoded = keras.utils.to_categorical(label)
            return label_encoded
        label = external_le.transform(self.df.Informativeness)
        label_encoded = keras.utils.to_categorical(label)
        return label_encoded

    def glove(self, embeddings_index, n_embedding):  # return GloVe embedding texts
        embedding_matrix = np.zeros(self.vocab_size, n_embedding)
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


def import_glove(dim=100):
    embeddings_index = dict()
    f = open('glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(str(dim)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index, dim