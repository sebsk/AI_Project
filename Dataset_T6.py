from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
import preprocessor as p
import pandas as pd
import numpy as np


class dataset_T6():
    def __init__(self, filename, preprocessing=True, emoji=False):
        self.df = pd.read_csv(filename)
        col_dict = {}
        for old_feature in self.df.columns.values:
            col_dict.update({old_feature: old_feature.replace(' ', '')})
        self.df.rename(columns=col_dict, inplace=True)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        label = [0] * self.df.shape[0]
        idx = self.df.index[self.df.label == 'on-topic'].tolist()
        for i in idx: label[i] = 1
        self.df['label'] = pd.Series(label)  # label
        word_collection = []
        if preprocessing:
            if emoji:
                emoji_re = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
                emojis = [regexp_tokenize(t, emoji_re) for t in self.df.tweet]
            p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
            all_tweets = [p.clean(t).lower() for t in self.df.tweet]
            tknzr = TweetTokenizer()
            all_tokens = [tknzr.tokenize(t) for t in all_tweets]
            en_stop = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            self.processed_texts = []  # preprocessed tweets
            if emoji:
                for i in range(len(all_tokens)):
                    self.processed_texts.append(' '.join([lemmatizer.lemmatize(t) for t in all_tokens[i] if t not in en_stop]+emojis[i]))
            else:
                for i in range(len(all_tokens)):
                    self.processed_texts.append(' '.join([lemmatizer.lemmatize(t) for t in all_tokens[i] if t not in en_stop]))
            for text in self.processed_texts:
                for word in text_to_word_sequence(text):
                    word_collection.append(word)
            self.vocab_size = len(set(word_collection))  # total_words
            self.tokenizer = Tokenizer()  # tokenizer
            self.tokenizer.fit_on_texts(self.processed_texts)
        else:
            for text in self.df.tweet:
                for word in text_to_word_sequence(text):
                    word_collection.append(word)
            self.vocab_size = len(set(word_collection))  # total_words
            self.tokenizer = Tokenizer()  # tokenizer
            self.tokenizer.fit_on_texts(self.df.tweet)

    def embedding(self, external_tokenizer=None):  # return embedding texts
        if external_tokenizer is None and self.tokenizer is None:
            return "please generate tokenizer first!"
        if external_tokenizer is None:
            try:
                encoded_tweets = self.tokenizer.texts_to_sequences(self.processed_texts)
            except:
                encoded_tweets = self.tokenizer.texts_to_sequences(self.df.tweet)
            padded_tweets = pad_sequences(encoded_tweets, maxlen=140, padding='post')
            return padded_tweets
        try:
            encoded_tweets = external_tokenizer.texts_to_sequences(self.processed_texts)
        except:
            encoded_tweets = external_tokenizer.texts_to_sequences(self.df.tweet)
        padded_tweets = pad_sequences(encoded_tweets, maxlen=140, padding='post')
        return padded_tweets

    def bow(self, m='binary', external_tokenizer=None):  # return vectorized texts, m can be binary, count, tfdif, freq
        if external_tokenizer is None and self.tokenizer is None:
            return "please generate tokenizer first!"
        if external_tokenizer is None:
            try:
                encoded_tweets = self.tokenizer.texts_to_matrix(self.processed_texts, mode=m)
            except:
                encoded_tweets = self.tokenizer.texts_to_matrix(self.df.tweet, mode=m)
            return encoded_tweets
        try:
            encoded_tweets = external_tokenizer.texts_to_matrix(self.processed_texts, mode=m)
        except:
            encoded_tweets = external_tokenizer.texts_to_matrix(self.df.tweet, mode=m)
        return encoded_tweets

    def glove(self, embedding_index, vtr_dim):  # return GloVe embedding texts
        embedding_matrix = np.zeros((self.vocab_size+1, vtr_dim))
        for word, i in self.tokenizer.word_index.iteritems():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model = Sequential()
        model.add(Embedding(self.vocab_size+1, output_dim=vtr_dim, weights=[embedding_matrix], trainable=False))
        model.compile('rmsprop', 'mse')
        try:
            encoded_tweets = np.asarray(pad_sequences(self.tokenizer.texts_to_sequences(self.processed_texts), maxlen=140, padding='post'))
        except:
            encoded_tweets = np.asarray(pad_sequences(self.tokenizer.texts_to_sequences(self.df.tweet), maxlen=140, padding='post'))
        embedding_texts = model.predict_on_batch(encoded_tweets)
        return embedding_texts

    def hashing_vectorizer(self, analyzer='word', ngram_range=(1,1), binary=False):  # for incremental learning
        h = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range, binary=binary)
        # analyzer: 'word', 'char', 'char_wb'; ngram_range:(min,max)
        try:
            encoded_tweets = h.transform(self.processed_texts)
        except:
            encoded_tweets = h.transform(self.df.tweet)
        return encoded_tweets

    def shuffle(self):  # shuffle dataframe
        self.df = self.df.sample(frac=1).reset_index(drop=True)

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