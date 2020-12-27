# pandas
import pandas as pd

# tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Concatenate, Attention, LayerNormalization, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# numpy
import numpy as np

# gensim
from gensim.models import KeyedVectors

# sklearn
from sklearn.model_selection import train_test_split

# matplotlib
import matplotlib.pyplot as plt

# typing
from typing import List, Tuple, Union, Any, Callable

# nltk
from nltk import word_tokenize, sent_tokenize

# alive progress
from alive_progress import alive_bar

# utils
import pickle
import os
import sys
import copy


def get_filepath(filename: str) -> str:
    """
    Get filepath of data. Or return the path of the data file on the data folder if the data folder exists.
    :param filename: The data csv file name.
    :return: The filepath of data csv file.
    """
    filepath = 'data/' + filename
    if os.getcwd().split('/')[-1] == 'body_to_title':
        filepath = '../' + filepath
    return filepath


def get_data(data_size: int = None, title='title', content='content'):
    """
    Get the data as a pandas dataframe.
    :param data_size: Max # of the data.
    :param title: The name of title column. It depends on the csv file.
    :param content: The name of content column. It depends on the csv file.
    :return: The data as a pandas dataframe.
    """
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
    """
    Split the text to the list of words.
    :param text: The text to split.
    :return: The list of words.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return word_tokenize(sentences[0])
    else:
        return sum([word_tokenize(sentence) for sentence in sentences], [])


def padding(word_list: List[str], max_len: int) -> List[str]:
    """
    Pad the list of words.
    For example, if it gets the list of words, [ 'I', 'am', 'a', 'student', '.' ],
    and the value of max_len is 10, its result will be
    [ 'I', 'am', 'a', 'student', '.', 'eot', 'eot', 'eot', 'eot', 'eot' ]
    :param word_list: The list of words to pad.
    :param max_len: The maximum length of the result. The length of result list will be max_len.
    :return: The padded list of words.
    """
    return word_list + ['eot'] * (max_len - len(word_list))


class Word2Vec:
    def __init__(self, filepath: str = get_filepath("GoogleNews-vectors-negative300.bin.gz")):
        try:
            self.vec = KeyedVectors.load_word2vec_format(filepath, binary=True)
        except FileNotFoundError:
            print('Vector file not found... please download vector file onto data folder!')
            print('wget "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"')
            print('filepath: ', filepath)
            exit()
        self.size = self.vec.vector_size

        self.SOT_VEC = self.vec['sot']
        self.EOT_VEC = tf.ones(self.size)
        self.OOV_VEC = tf.zeros(self.size)

        self.vec.add(['eot'], [self.EOT_VEC])

    def __getitem__(self, item: str):
        return self.vec[item] if item in self.vec else self.OOV_VEC

    def vector_to_word(self, vector: Union[np.ndarray, tf.Tensor]) -> str:
        return self.vec.similar_by_vector(vector)[0][0]


if '-l' in sys.argv:
    print('Loading lite vec...')
    with open(get_filepath('word2vec_lite.pickle'), 'rb') as f:
        word2vec = pickle.load(f)
else:
    print('Loading word2vec...')
    word2vec = Word2Vec()
    lite_vec = copy.copy(word2vec)
    lite_vec.vec = None
    with open(get_filepath('word2vec_lite.pickle'), 'wb') as f:
        pickle.dump(lite_vec, f)
    print('Lite vector saving done!')


def word_to_vector(word_list: List[str]) -> np.ndarray:
    return np.array([word2vec[word] for word in word_list])


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
    return tf.reduce_any(tf.math.not_equal(input_tensor, word2vec.EOT_VEC), -1)


def multi_head_attention(query: tf.Tensor, value: tf.Tensor,
                         query_mask: tf.Tensor = None, value_mask: tf.Tensor = None) -> tf.Tensor:
    len_query = query.shape[-2]
    len_value = value.shape[-2]

    embedding_dim = query.shape[-1]
    num_heads = 12
    while num_heads > 1:
        if embedding_dim % num_heads == 0:
            break
        num_heads -= 1

    query_separate_layer = Reshape((len_query, num_heads, -1))
    value_separate_layer = Reshape((len_value, num_heads, -1))

    query_separated = query_separate_layer(query)
    value_separated = value_separate_layer(value)

    output_list: List[tf.Tensor] = []

    query_reshape_layer = Reshape((len_query, -1))
    value_reshape_layer = Reshape((len_value, -1))

    for i in range(num_heads):
        query_input = query_reshape_layer(query_separated[:, :, i])
        value_input = value_reshape_layer(value_separated[:, :, i])

        attention_layer = Attention(use_scale=True)
        attention_output = attention_layer(
            [query_input, value_input],
            mask=[query_mask, value_mask]
        )
        output_list.append(attention_output)

    concat_layer = Concatenate(axis=-1)
    output = concat_layer(output_list)
    return output


def add_and_normalization(input_tensor: tf.Tensor, adding_tensor: tf.Tensor,
                          epsilon: float = 1e-6) -> tf.Tensor:
    added_tensor = input_tensor + adding_tensor
    norm_layer = LayerNormalization(epsilon=epsilon)
    output = norm_layer(added_tensor)
    return output


def feed_forward(input_tensor: tf.Tensor, num_hidden: int) -> tf.Tensor:
    embedding_dim = input_tensor.shape[-1]

    relu_layer = Dense(num_hidden, activation=relu)
    relu_output = relu_layer(input_tensor)

    output_layer = Dense(embedding_dim)
    output = output_layer(relu_output)

    return output


def encoder_layer(input_tensor: tf.Tensor, num_ff_hidden: int,
                  mask: tf.Tensor = None) -> tf.Tensor:
    attention_output = multi_head_attention(
        input_tensor, input_tensor, query_mask=mask, value_mask=mask
    )
    output = add_and_normalization(input_tensor, attention_output)

    ff_output = feed_forward(output, num_ff_hidden)
    output = add_and_normalization(output, ff_output)

    return output


def decoder_layer(dec_input: tf.Tensor, enc_output: tf.Tensor, num_ff_hidden: int,
                  enc_mask: tf.Tensor = None, dec_mask: tf.Tensor = None) -> tf.Tensor:
    self_attention_output = multi_head_attention(
        dec_input, dec_input, query_mask=dec_mask, value_mask=dec_mask
    )
    output = add_and_normalization(dec_input, self_attention_output)

    attention_output = multi_head_attention(
        output, enc_output, query_mask=dec_mask, value_mask=enc_mask
    )
    output = add_and_normalization(output, attention_output)

    ff_output = feed_forward(output, num_ff_hidden)
    output = add_and_normalization(output, ff_output)

    return output


class Encoder:
    def __init__(self, num_layers: int, num_ff_hidden: int):
        self.num_layers = num_layers
        self.num_ff_hidden = num_ff_hidden

    def __call__(self, input_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mask = create_padding_mask(input_tensor)
        output = positional_encoding(input_tensor)
        for i in range(self.num_layers):
            output = encoder_layer(output, self.num_ff_hidden, mask)
        return output, mask


class Decoder:
    def __init__(self, num_layers: int, num_ff_hidden: int):
        self.num_layers = num_layers
        self.num_ff_hidden = num_ff_hidden

    def __call__(self, dec_input: tf.Tensor, enc_output: tf.Tensor,
                 enc_mask: tf.Tensor) -> tf.Tensor:
        dec_mask = create_padding_mask(dec_input)
        output = positional_encoding(dec_input)
        for i in range(self.num_layers):
            output = decoder_layer(
                output, enc_output, self.num_ff_hidden, enc_mask=enc_mask, dec_mask=dec_mask
            )
        return output


class Transformer(Model):
    def __init__(self, num_layers: int, num_ff_hidden: int,
                 input_shape: Tuple[int, int], output_shape: Tuple[int, int]):
        self.num_layers = num_layers
        self.num_ff_hidden = num_ff_hidden

        self.len_input = input_shape[-2]
        self.len_output = output_shape[-2]
        self.embedding_dim = input_shape[-1]
        if self.embedding_dim != output_shape[-1]:
            assert 'input embedding dimension and output embedding dimension must be same'

        self.encoder = Encoder(self.num_layers, self.num_ff_hidden)
        self.decoder = Decoder(self.num_layers, self.num_ff_hidden)

        encoder_input = Input(
            shape=(self.len_input, self.embedding_dim),
            name='input_layer'
        )

        decoder_input = Input(
            shape=(self.len_output - 1, self.embedding_dim),
            name='output_layer'
        )

        output = self.execute(encoder_input, decoder_input)

        super().__init__(
            inputs=[encoder_input, decoder_input],
            outputs=output,
            name='transformer_functional_model'
        )

    def execute(self, enc_input: tf.Tensor, dec_input: tf.Tensor) -> tf.Tensor:
        encoder_output, enc_mask = self.encoder(enc_input)
        decoder_output = self.decoder(dec_input, encoder_output, enc_mask)

        final_layer = Dense(self.embedding_dim)
        output = final_layer(decoder_output)
        return output

    class CustomSchedule(LearningRateSchedule):
        def __init__(self, d_model, warm_up_steps=4000, name='transformer_custom_schedule'):
            super(Transformer.CustomSchedule, self).__init__()

            self.d_model = tf.cast(d_model, tf.float32)
            self.warm_up_steps = warm_up_steps

            self.name = name

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warm_up_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            return {
                "name": self.name
            }

    def compile(self, optimizer: Optimizer = None, loss: Loss = None,
                **kwargs: Any):
        if optimizer is None:
            learning_rate = Transformer.CustomSchedule(self.embedding_dim)
            optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        if loss is None:
            loss = MeanSquaredError()
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

    @staticmethod
    def __encoder_decoder_data_split(x_data: np.ndarray, y_data: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x_data, y_data[:, :-1], y_data[:, 1:]

    def fit(self, x: np.ndarray = None, y: np.ndarray = None,
            batch_size: int = None, epochs: int = 1,
            validation_data: Tuple[np.ndarray, np.ndarray] = None, **kwargs):
        x_test, y_test = validation_data
        x_enc_train, x_dec_train, y_train = Transformer.__encoder_decoder_data_split(x, y)
        x_enc_test, x_dec_test, y_test = Transformer.__encoder_decoder_data_split(x_test, y_test)
        return super().fit(
            [x_enc_train, x_dec_train], y_train,
            batch_size, epochs,
            validation_data=([x_enc_test, x_dec_test], y_test),
            **kwargs
        )

    def __get_initial_sequence(self) -> tf.Tensor:
        text_list = ['sot'] + ['eot'] * (self.len_output - 2)
        text_sequence = word_to_vector(text_list)
        return tf.constant([text_sequence])

    def predict(self, text: str, *args, **kwargs) -> str:
        text_split = split_text(text)
        text_padded = padding(text_split, self.len_input)
        text_sequence = word_to_vector(text_padded)
        inputs = tf.constant([text_sequence], dtype=tf.float32)

        enc_output, enc_mask = self.encoder(inputs)

        now = 'sot'
        dec_input = self.__get_initial_sequence()
        output: List[str] = []
        idx = 0
        while now != 'eot' and idx < self.len_output:
            dec_output = self.decoder(dec_input, enc_output, enc_mask)

            idx = len(output)
            vec = dec_output[0, idx].numpy()
            now = word2vec.vector_to_word(vec)
            output.append(now)

            dec_input = tf.concat([
                tf.reshape(dec_input[0, :(idx + 1)], (1, -1, self.embedding_dim)),
                tf.reshape(dec_output[0, idx], (1, -1, self.embedding_dim)),
                tf.reshape(dec_input[0, (idx + 2):], (1, -1, self.embedding_dim))
            ], axis=1)

        return ' '.join(output)


def assign(func: Callable, array: List[Any], *args, title: str = None) -> Any:
    """
    Return the list of the results of the func run with the each item of array and args as a parameter.
    :param func: The function to run.
    :param array: The array that will be passed as a parameter of the func.
    :param args: The arguments that will be passed as a parameter of the func additionally.
    :param title: The title of the alive progress bar.
    :return: The list of the results of the func execution.
    """
    result = []
    with alive_bar(len(array), title=title) as bar:
        for item in array:
            result.append(func(item, *args))
            bar()
    return result


if __name__ == '__main__':
    is_debugging_mode = sys.gettrace()
    if is_debugging_mode:
        print('========== DEBUGGING MODE ==========')
        DATA_SIZE = 5

        NUM_LAYERS = 1
        NUM_FF_HIDDEN = 512
        BATCH_SIZE = 1
        EPOCHS = 20

        data_filename = 'dummy'
    else:
        DATA_SIZE = 10000

        NUM_LAYERS = 6
        NUM_FF_HIDDEN = 1024
        BATCH_SIZE = 1
        EPOCHS = 20

        data_filename = get_filepath('learning_data_%d.pickle' % DATA_SIZE)
    checkpoint_path = 'checkpoint/model_checkpoint.ckpt'

    try:
        with open(data_filename, 'rb') as f_data:
            print('file found:', data_filename)

            X_train, X_test, Y_train, Y_test = train_test_split(*pickle.load(f_data))

            max_word_title = Y_train.shape[1]
            max_word_content = X_train.shape[1]
    except FileNotFoundError:
        # Import Data
        print('Importing data...')
        titles, contents = get_data(DATA_SIZE, content='summary')

        # Pre-Processing
        title_split = assign(split_text, titles, title='[Split title data]')
        content_split = assign(split_text, contents, title='[Split content data]')

        max_word_title = max([len(title) for title in title_split])
        max_word_content = max([len(content) for content in content_split])

        title_padded = assign(padding, title_split, max_word_title, title='[Padding title data]')
        content_padded = assign(padding, content_split, max_word_content, title='[Padding content data]')

        title_sequences = assign(word_to_vector, title_padded, title='[Sequencing title data]')
        content_sequences = assign(word_to_vector, content_padded, title='[Sequencing content data]')

        X_data = np.array(content_sequences)
        Y_data = np.array(title_sequences)

        print('Saving data...')
        if data_filename != 'dummy':
            with open(data_filename, 'wb') as f:
                pickle.dump((X_data, Y_data), f)

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)

    print('Loading model...')
    model = Transformer(
        NUM_LAYERS, NUM_FF_HIDDEN,
        input_shape=(max_word_content, word2vec.size),
        output_shape=(max_word_title, word2vec.size)
    )

    print('Compiling...')
    model.compile()

    print(model.count_params())

    print('Fitting...')
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping, checkpoint]
    )

    with open('history.pickle', 'wb') as f_history:
        pickle.dump(history.history, f_history)
    model.save('transformer_model.h5')

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    TEST_CONTENT = r'''"The incoming Trump administration could choose to no longer defend the executive branch against the suit, which challenges the administration’s authority to spend billions of dollars on health insurance subsidies for   and   Americans, handing House Republicans a big victory on    issues.
    Collyer ruled that House Republicans had the standing to sue the executive branch over a spending dispute and that the Obama administration had been distributing the health insurance subsidies, in violation of the Constitution, without approval from Congress.
    Anticipating that the Trump administration might not be inclined to mount a vigorous fight against the House Republicans given the  ’s dim view of the health care law, a team of lawyers this month sought to intervene in the case on behalf of two participants in the health care program.
    ” No matter what happens, House Republicans say, they want to prevail on two overarching concepts: the congressional power of the purse, and the right of Congress to sue the executive branch if it violates the Constitution regarding that spending power.
    Just as important to House Republicans, Judge Collyer found that Congress had the standing to sue the White House on this issue  —   a ruling that many legal experts said was flawed  —   and they want that precedent to be set to restore congressional leverage over the executive branch.
    But on spending power and standing, the Trump administration may come under pressure from advocates of presidential authority to fight the House no matter their shared views on health care, since those precedents could have broad repercussions."'''

    print('Predict data...')
    print(TEST_CONTENT)
    print(model.predict(TEST_CONTENT))
