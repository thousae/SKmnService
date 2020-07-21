from .statics import static_vars
from nltk.tokenize import word_tokenize, sent_tokenize
import re


@static_vars(SOT='sot', EOT='eot', PAD=0, _Preprocessor__okt=None)
class Preprocessor:
    def __init__(self):
        self.__is_hangul_data = False
        self.__max_words = None
        self.__tokenizer = None

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
                return sep if sep_by_sentence else return sum(sep, [])
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
            return separated

    def fit_tokenizer(self, sequence):
        if self.__tokenizer is None:
            from tensorflow.keras.preprocessing.text import Tokenizer
            self.__tokenizer = Tokenizer()
            self.__tokenizer.word_index[Preprocessor.PAD] = Preprocessor.PAD
            self.__tokenizer.index_word[Preprocessor.PAD] = Preprocessor.PAD
        self.__tokenizer.fit_on_texts(sequence)
        return self

    def tokenize(self, sequence):
        return self.__tokenizer.texts_to_sequences(sequence)

    @property
    def max_words(self):
        if self.__max_words is None:
            raise ValueError("You have to compile your text before get max words")
        return self.__max_words

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
