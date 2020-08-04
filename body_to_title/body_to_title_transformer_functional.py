# pandas
import pandas as pd

# tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Attention, LayerNormalization, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import MeanSquaredError

# numpy
import numpy as np

# gensim
from gensim.models import KeyedVectors

# sklearn
from sklearn.model_selection import train_test_split

# matplotlib
import matplotlib.pyplot as plt

# typing
from typing import List, Tuple

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


def encoder_decoder_data_split(x_data: np.ndarray, y_data: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    return x_train, x_test, y_train[:, :-1], y_test[:, :-1], y_train[:, 1:], y_test[:, 1:]


DATA_SIZE = 5

# data_filename = get_filepath('learning_data_%d.pickle' % DATA_SIZE)
data_filename = 'dummy'
try:
    with open(data_filename, 'rb') as f_data:
        print('file found:', data_filename)

        X_enc_train, X_enc_test, X_dec_train, X_dec_test, Y_train, Y_test \
            = encoder_decoder_data_split(*pickle.load(f_data))

        max_word_title = Y_train.shape[1] + 1
        max_word_content = X_enc_test.shape[1]
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

    if data_filename != 'dummy':
        with open(data_filename, 'wb') as f:
            pickle.dump((X_data, Y_data), f)

    X_enc_train, X_enc_test, X_dec_train, X_dec_test, Y_train, Y_test \
        = encoder_decoder_data_split(X_data, Y_data)


# Training Model
NUM_HEADS = 3
NUM_LAYERS = 1
NUM_FF_HIDDEN = 512
BATCH_SIZE = 1
EPOCHS = 20

# Setting
# tf.config.experimental_run_functions_eagerly(True)


def positional_encoding(input_tensor: tf.Tensor, scale=10000) -> tf.Tensor:
    input_dim = input_tensor.shape[-2]
    dim_model = input_tensor.shape[-1]

    pos = np.arange(input_dim)[:, np.newaxis]
    i = np.arange(dim_model)[np.newaxis, :]
    i[:, 1::2] = i[:, 1::2] - 1

    pos_encoder = pos / np.power(scale, i / np.float32(dim_model))
    pos_encoder[:, 0::2] = np.sin(pos_encoder[:, 0::2])
    pos_encoder[:, 1::2] = np.cos(pos_encoder[:, 1::2])

    return input_tensor + pos_encoder


def create_padding_mask(input_tensor: tf.Tensor) -> tf.Tensor:
    return tf.reduce_any(tf.math.not_equal(input_tensor, EOT_VEC), -1)


def multi_head_attention(query: tf.Tensor, value: tf.Tensor,
                         query_mask: tf.Tensor = None, value_mask: tf.Tensor = None) -> tf.Tensor:
    len_query = query.shape[-2]
    len_value = value.shape[-2]

    query_separated = tf.reshape(query, (BATCH_SIZE, len_query, NUM_HEADS, -1))
    value_separated = tf.reshape(value, (BATCH_SIZE, len_value, NUM_HEADS, -1))

    output_list: List[tf.Tensor] = []

    for i in range(NUM_HEADS):
        query_input = tf.reshape(query_separated[:, :, i], (BATCH_SIZE, len_query, -1))
        value_input = tf.reshape(value_separated[:, :, i], (BATCH_SIZE, len_value, -1))

        attention_layer = Attention(use_scale=True)
        attention_output = attention_layer(
            [query_input, value_input],
            mask=[query_mask, value_mask]
        )
        output_list.append(attention_output)

    output = tf.concat(output_list, axis=-1)
    return output


def add_and_normalization(input_tensor: tf.Tensor, adding_tensor: tf.Tensor,
                          epsilon: float = 1e-6) -> tf.Tensor:
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
    attention_output = multi_head_attention(
        input_tensor, input_tensor, query_mask=mask, value_mask=mask
    )
    output = add_and_normalization(input_tensor, attention_output)

    ff_output = feed_forward(output)
    output = add_and_normalization(output, ff_output)

    return output


def decoder_layer(dec_input: tf.Tensor, enc_output: tf.Tensor,
                  enc_mask: tf.Tensor = None, dec_mask: tf.Tensor = None) -> tf.Tensor:
    self_attention_output = multi_head_attention(
        dec_input, dec_input, query_mask=dec_mask, value_mask=dec_mask
    )
    output = add_and_normalization(dec_input, self_attention_output)

    attention_output = multi_head_attention(
        output, enc_output, query_mask=dec_mask, value_mask=enc_mask
    )
    output = add_and_normalization(output, attention_output)

    ff_output = feed_forward(output)
    output = add_and_normalization(output, ff_output)

    return output


def encoder(input_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    mask = create_padding_mask(input_tensor)
    output = positional_encoding(input_tensor)
    for _ in range(NUM_LAYERS):
        output = encoder_layer(output, mask)
    return output, mask


def decoder(dec_input: tf.Tensor, enc_output: tf.Tensor, enc_mask: tf.Tensor) -> tf.Tensor:
    dec_mask = create_padding_mask(dec_input)
    output = positional_encoding(dec_input)
    for _ in range(NUM_LAYERS):
        output = decoder_layer(output, enc_output, enc_mask=enc_mask, dec_mask=dec_mask)
    return output


def transformer(enc_input: tf.Tensor, dec_input: tf.Tensor) -> tf.Tensor:
    encoder_output, enc_mask = encoder(enc_input)
    decoder_output = decoder(dec_input, encoder_output, enc_mask)

    final_layer = Dense(embedding_dim)
    output = final_layer(decoder_output)
    return output


encoder_input = Input(
    shape=(max_word_content, embedding_dim),
    batch_size=BATCH_SIZE,
    name='input_layer'
)

decoder_input = Input(
    shape=(max_word_title - 1, embedding_dim),
    batch_size=BATCH_SIZE,
    name='output_layer'
)

transformer_training_output = transformer(encoder_input, decoder_input)

model = Model(
    inputs=[encoder_input, decoder_input],
    outputs=transformer_training_output,
    name='transformer_functional_model'
)


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warm_up_steps=4000, name='transformer_custom_schedule'):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warm_up_steps = warm_up_steps

        self.name = name

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            # "initial_learning_rate": self.initial_learning_rate,
            # "decay_steps": self.decay_steps,
            # "decay_rate": self.decay_rate,
            # "staircase": self.staircase,
            "d_model": self.d_model,
            "warm_up_steps": self.warm_up_steps,
            "name": self.name
        }


learning_rate = CustomSchedule(embedding_dim)
model.compile(
    optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
    loss=MeanSquaredError()
)

print(model.count_params())

history = model.fit(
    [X_enc_train, X_dec_train], Y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=([X_enc_test, X_dec_test], Y_test),
)
with open('history.pickle', 'wb') as f_history:
    pickle.dump(history.history, f_history)
model.save('transformer_model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


def get_initial_sequence() -> tf.Tensor:
    text_list = ['sot'] + ['eot'] * (max_word_title - 2)
    text_sequence = word_to_vector(text_list)
    return tf.constant([text_sequence])


def predict(text: str) -> str:
    text_split = split_text(text)
    text_padded = padding(text_split, max_word_content)
    text_sequence = word_to_vector(text_padded)
    inputs = tf.constant([text_sequence], dtype=tf.float32)

    enc_output, enc_mask = encoder(inputs)

    now = 'sot'
    dec_input = get_initial_sequence()
    output: List[str] = []
    idx = 0
    while now != 'eot' and idx < max_word_title:
        dec_output = decoder(dec_input, enc_output, enc_mask)

        idx = len(output)
        vec = dec_output[0, idx].numpy()
        now = word2vec.similar_by_vector(vec)[0][0]
        output.append(now)

        # dec_input[0, idx + 1] = dec_output[0, idx]
        calculation = tf.concat([dec_input[0, :(idx + 1)], dec_output[0, idx], dec_input[0, (idx + 2):]])
        dec_input = calculation

    return ' '.join(output)


print(contents[0])
print(predict(contents[0]))
