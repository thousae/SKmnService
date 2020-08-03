# pandas
from abc import ABC

import pandas as pd

# tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Concatenate, Attention, LayerNormalization, Dense
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# numpy
import numpy as np

# gensim
from gensim.models import KeyedVectors

# sklearn
from sklearn.model_selection import train_test_split

# matplotlib
import matplotlib.pyplot as plt

# typing
from typing import List

# utils
from nltk import word_tokenize, sent_tokenize
import pickle
import os


def get_filepath(filename: str) -> str:
    filepath = 'data/' + filename
    if os.getcwd().split('/')[-1] == 'body_to_title':
        filepath = '../' + filepath
    return filepath


def get_data(data_size: int = None, title='title', content='content'):
    csv_filename = get_filepath('articles.csv')

    with open(csv_filename) as f_csv:
        articles = pd.read_csv(f_csv)

    articles = articles.dropna()

    articles[title] = 'sot ' + articles[title] + ' eot'

    if data_size is not None:
        articles = articles[:data_size]

    _titles = articles[title].tolist()
    _contents = articles[content].tolist()

    return _titles, _contents


def split_text(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return word_tokenize(sentences[0])
    else:
        return sum([word_tokenize(sentence) for sentence in sentences], [])


word2vec = KeyedVectors.load_word2vec_format("../../GoogleNews-vectors-negative300.bin", binary=True)
embedding_dim = word2vec['hello'].shape[0]

SOT_VEC = word2vec['sot']
EOT_VEC = np.ones(embedding_dim)
OOV_VEC = np.zeros(embedding_dim)

word2vec.add(['eot'], [EOT_VEC])
def word_to_vector(word_list: List[str]) -> np.ndarray:
    return np.array([word2vec[word] if word in word2vec else OOV_VEC
                     for word in word_list])


def padding(word_list: List[str], max_len: int) -> List[str]:
    return word_list + ['eot'] * (max_len - len(word_list))


DATA_SIZE = 5

data_filename = get_filepath('learning_data_%d.pickle' % DATA_SIZE)
try:
    with open(data_filename, 'rb') as f_data:
        print('file found:', data_filename)

        learning_data = pickle.load(f_data)
        X_train, X_test, Y_train, Y_test = learning_data

        max_word_title = Y_train.shape[1]
        max_word_content = X_train.shape[1]
except FileNotFoundError:
    # Import Data
    titles, contents = get_data(DATA_SIZE, content='summary')

    # Pre-Processing
    title_split = [split_text(title) for title in titles]
    content_split = [split_text(content) for content in contents]

    max_word_title = max([len(title) for title in title_split])
    max_word_content = max([len(content) for content in content_split])

    title_padded = [padding(title, max_word_title) for title in title_split]
    content_padded = [padding(content, max_word_content) for content in content_split]

    title_sequences = [word_to_vector(title) for title in title_padded]
    content_sequences = [word_to_vector(content) for content in content_padded]

    X_data = np.array(content_sequences)
    Y_data = np.array(title_sequences)

    learning_data = train_test_split(X_data, Y_data)
    with open(data_filename, 'wb') as f:
        pickle.dump(learning_data, f)
    X_train, X_test, Y_train, Y_test = learning_data


# Training Model
NUM_HEADS = 8
NUM_LAYERS = 6
NUM_FF_HIDDEN = 512
BATCH_SIZE = 1
EPOCHS = 20


def positional_encoding(input_tensor: tf.Tensor, scale=10000) -> tf.Tensor:
    input_dim = input_tensor.shape[0]
    dim_model = input_tensor.shape[1]

    pos = np.arange(input_dim)[:, np.newaxis]
    i = np.arange(dim_model)[np.newaxis, :]
    i[:, 1::2] = i[:, 1::2] - 1

    encoder = pos / np.power(scale, i / np.float32(dim_model))
    encoder[:, 0::2] = np.sin(encoder[:, 0::2])
    encoder[:, 1::2] = np.cos(encoder[:, 1::2])

    return input_tensor + encoder


def create_padding_mask(input_tensor: tf.Tensor) -> tf.Tensor:
    return tf.reduce_all(tf.math.equal(input_tensor, EOT_VEC), -1)


def multi_head_attention(query: tf.Tensor, value: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
    separate_layer = Reshape((NUM_HEADS, -1, embedding_dim))

    query_separated = separate_layer(query)
    value_separated = separate_layer(value)

    output_list: List[tf.Tensor] = []
    reshape_layer = Reshape((-1, embedding_dim))
    concat_layer = Concatenate(axis=-2)

    for i in range(NUM_HEADS):
        query_input = reshape_layer(query_separated[:, i])
        value_input = reshape_layer(value_separated[:, i])
        attention_layer = Attention(
            use_scale=True,
            name='masked_' if mask is not None else '' + 'multi_head_attention_%d' % (i + 1)
        )
        attention_output = attention_layer([query_input, value_input], mask=mask)
        output_list.append(attention_output)
    output = concat_layer(output_list)
    return output


def add_and_normalization(input_tensor: tf.Tensor, adding_tensor: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
    added_tensor = input_tensor + adding_tensor
    norm_layer = LayerNormalization(epsilon=epsilon, name='normalization_layer')
    output = norm_layer(added_tensor)
    return output


def feed_forward(input_tensor: tf.Tensor) -> tf.Tensor:
    relu_layer = Dense(NUM_FF_HIDDEN, activation=relu, name='feed_forward_layer_1')
    relu_output = relu_layer(input_tensor)

    output_layer = Dense(embedding_dim, name='feed_forward_layer_2')
    output = output_layer(relu_output)

    return output


def encoder_layer(input_tensor: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
    attention_output = multi_head_attention(input_tensor, input_tensor, mask)
    output = add_and_normalization(input_tensor, attention_output)

    ff_output = feed_forward(output)
    output = add_and_normalization(output, ff_output)

    return output


def decoder_layer(dec_input: tf.Tensor, enc_output: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
    self_attention_output = multi_head_attention(dec_input, dec_input, mask)
    output = add_and_normalization(dec_input, self_attention_output)

    attention_output = multi_head_attention(output, enc_output, mask)
    output = add_and_normalization(output, attention_output)

    ff_output = feed_forward(output)
    output = add_and_normalization(output, ff_output)

    return output


def encoder(input_tensor: tf.Tensor) -> tf.Tensor:
    mask = create_padding_mask(input_tensor)
    output = positional_encoding(input_tensor)
    for _ in range(NUM_LAYERS):
        output = encoder_layer(output, mask)
    return output


def decoder(dec_input: tf.Tensor, enc_output: tf.Tensor) -> tf.Tensor:
    mask = create_padding_mask(dec_input)
    output = positional_encoding(dec_input)
    for _ in range(NUM_LAYERS):
        output = decoder_layer(output, enc_output, mask)
    return output


def transformer(enc_input: tf.Tensor, dec_input: tf.Tensor) -> tf.Tensor:
    encoder_output = encoder(enc_input)
    decoder_output = decoder(dec_input, encoder_output)
    return decoder_output


def output_concatenate(dec_input: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    concatenate = Concatenate(axis=-2)
    return concatenate(dec_input[:, 0], output)


encoder_input = Input(
    shape=(max_word_content, embedding_dim),
    batch_size=BATCH_SIZE,
    name='input_layer'
)

decoder_input = Input(
    shape=(max_word_title, embedding_dim),
    batch_size=BATCH_SIZE,
    name='output_layer'
)

transformer_training_output = output_concatenate(
    decoder_input,
    transformer(encoder_input, decoder_input[:, :-1, ...])
)

model = Model(
    inputs=[encoder_input, decoder_input],
    outputs=transformer_training_output,
    name='transformer_functional_model'
)


class CustomSchedule(LearningRateSchedule, ABC):
    def __init__(self, d_model, warm_up_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warm_up_steps = warm_up_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(embedding_dim)
model.compile(
    optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
)

history = model.fit(
    [X_train, Y_train], Y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=([X_test, Y_test], Y_test)
)
with open('history.pickle', 'wb') as f_history:
    pickle.dump(history.history, f_history)
model.save('transformer_model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


def get_initial_sequence() -> tf.Tensor:
    text_list = ['sot'] + ['eot'] * (max_word_content - 2)
    text_sequence = word_to_vector(text_list)
    return tf.constant(text_sequence)


def predict(text: str) -> str:
    text_split = split_text(text)
    text_padded = padding(text_split, max_word_content)
    text_sequence = word_to_vector(text_padded)
    inputs = tf.constant(text_sequence)

    enc_output = encoder(inputs)

    now = 'sot'
    dec_input = get_initial_sequence()
    output: List[str] = []
    while now != 'eot':
        dec_output = decoder(dec_input, enc_output)

        idx = len(output)
        now = word2vec.similar_by_vector(dec_output[idx].numpy())
        output.append(now)

        dec_input[idx + 1] = dec_output[idx]

    return ' '.join(output)
