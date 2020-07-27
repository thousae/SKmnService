from .statics import static_vars
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import numpy as np


@static_vars(SOT='sot', EOT='eot', PAD=0, _Preprocessor__okt=None, _Preprocessor__tokenizer=None)
class Preprocessor:
    def __init__(self):
        self.__is_hangul_data = False
        self.__sequence_len = 0
        self.__max_words = None

    @staticmethod
    def __is_hangul(text):
        return re.match('.*[가-힣ㄱ-ㅎㅏ-ㅣ].*', text) is not None

    @staticmethod
    def __add_sot_token(text):
        return Preprocessor.SOT + ' ' + text

    @staticmethod
    def __add_eot_token(text):
        return text + ' ' + Preprocessor.EOT

    def __separate(self, text, sep_by_sentence=False):
        if self.__is_hangul_data:
            if Preprocessor.__okt is None:
                from konlpy.tag import Okt
                Preprocessor.__okt = Okt()
            return Preprocessor.__okt.morphs(text)
        else:
            def tok(t):
                sep = [word_tokenize(s) for s in sent_tokenize(t)]
                return sep if sep_by_sentence else sum(sep, [])
            try:
                return tok(text)
            except LookupError:
                import nltk
                nltk.download('punkt')
                return tok(text)

    def is_hangul_data(self, toggle=True):
        self.__is_hangul_data = toggle
        return self

    def separate_to_words(self, target, add_token=False, sep_by_sentence=False):
        if isinstance(target, list):
            return [self.separate_to_words(elem) for elem in target]
        else:
            if add_token:
                target = Preprocessor.__add_sot_token(target)
                target = Preprocessor.__add_eot_token(target)
            separated = self.__separate(target, sep_by_sentence=sep_by_sentence)
            self.__sequence_len = max(self.__sequence_len, len(separated))
            return separated

    def padding(self, target):
        if not isinstance(target, list):
            raise ValueError('The target must be list')
        if isinstance(target[0], list):
            return [self.padding(elem) for elem in target]
        return target + [Preprocessor.PAD] * (self.__sequence_len - len(target))

    @staticmethod
    def fit_tokenizer(sequence):
        if Preprocessor.__tokenizer is None:
            from tensorflow.keras.preprocessing.text import Tokenizer
            Preprocessor.__tokenizer = Tokenizer()
            Preprocessor.__tokenizer.word_index[Preprocessor.PAD] = Preprocessor.PAD
            Preprocessor.__tokenizer.index_word[Preprocessor.PAD] = Preprocessor.PAD
        Preprocessor.__tokenizer.fit_on_texts(sequence)

    def tokenize(self, sequence):
        result = self.__tokenizer.texts_to_sequences(sequence)
        result.to_numpy = lambda value: np.array(value)
        return result

    def get_sequence_len(self):
        return self.__sequence_len

    @staticmethod
    def get_num_words():
        if Preprocessor.__tokenizer is None:
            raise ValueError("You have to compile your text before get the number of the words")
        return len(Preprocessor.__tokenizer.word_counts) + 1

    @property
    def tokenizer(self):
        if self.__tokenizer is None:
            raise ValueError("You have to compile your text before get max words")
        return self.__tokenizer


if __name__ == '__main__':
    import pandas as pd

    with open('data/articles.csv') as f:
        articles = pd.read_csv(f)
    print(articles.head())

    titles = articles['title'].tolist()
    contents = articles['content'].tolist()

    preprocessor = Preprocessor()

    content_separated = preprocessor.separate_to_words(contents)
    title_separated = preprocessor.separate_to_words(titles, add_token=True)

    preprocessor.fit_tokenizer(content_separated)
    preprocessor.fit_tokenizer(title_separated)

    content_sequence = preprocessor.tokenize(content_separated)
    title_sequence = preprocessor.tokenize(title_separated)

    print('success')
