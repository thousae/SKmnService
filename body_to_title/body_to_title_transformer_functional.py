# pandas
import pandas as pd

# tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Attention, Concatenate, Dense
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# numpy
import numpy as np

# gensim
from gensim.models import KeyedVectors

# utils
from nltk import word_tokenize, sent_tokenize

# sklearn
from sklearn.model_selection import train_test_split

# typing
from typing import List, Tuple

# Import Data
def get_data(data_size: int = None, title='title', content='content'):
    try:
        with open('data/articles.csv') as f:
            articles = pd.read_csv(f)
    except FileNotFoundError:
        with open('../data/articles.csv') as f:
            articles = pd.read_csv(f)

    if data_size is not None:
        articles = articles[:data_size]

    _titles = articles[title].tolist()
    _contents = articles[content].tolist()

    return _titles, _contents

titles, contents = get_data(5, content='summary')

# Pre-processing
word2vec = KeyedVectors.load_word2vec_format("word2vec/GoogleNews-vectors-negative300.bin", binary=True)
embedding_dim = word2vec['hello'].shape[0]

print(embedding_dim)

def split_text(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return word_tokenize(sentences[0])
    else:
        return sum([word_tokenize(sentence) for sentence in sentences], [])

def embedding(word_list: List[str]) -> List[np.ndarray]:
    return [word2vec[word]
            if word in word2vec else np.zeros(embedding_dim)
            for word in word_list]

def positional_encoding(sequence: np.ndarray, scale=10000) -> np.ndarray:
    input_dim = sequence.shape[0]
    dim_model = sequence.shape[1]

    pos = tf.range(input_dim)[:, tf.newaxis]
    i = tf.range(dim_model)[tf.newaxis, :]
    i[:, 1::2] = i[:, 1::2] - 1

    encoder = pos / tf.pow(scale, i / tf.float32(dim_model))
    encoder[:, 0::2] = tf.sin(encoder[:, 0::2])
    encoder[:, 1::2] = tf.cos(encoder[:, 1::2])

    return sequence + encoder

def pre_processing(word_list: List[str], max_len: int) -> np.ndarray:
    sequence_list = embedding(word_list)
    sequence_list_padded = pad_sequences(sequence_list, max_len)
    sequence_array_padded = np.array(sequence_list_padded)
    sequence_encoded = positional_encoding(sequence_array_padded)
    return sequence_encoded

titles_split = [split_text(title) for title in titles]
contents_split = [split_text(content) for content in contents]

max_word_title = max([len(title) for title in titles_split])
max_word_content = max([len(content) for content in contents_split])

title_sequences = [pre_processing(title, max_word_title) for title in titles_split]
content_sequences = [pre_processing(contents, max_word_content) for content in contents_split]

X_train, X_test, Y_train, Y_test = train_test_split(content_sequences, title_sequences)
