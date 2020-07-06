class ProgressBar:
    def __init__(self, data_size, length):
        self.__data_size = data_size
        self.__length = length
        self.__printing_indexes = [int(data_size / length * i) for i in range(length + 1)]

    def get_progress_bar(self, i):
        try:
            cur = self.__printing_indexes.index(i)
            return '[' + '=' * cur + '>' + ' ' * (self.__length - cur) + ']'
        except ValueError:
            return None
