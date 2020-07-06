import requests
import re
from bs4 import BeautifulSoup
from typing import List, Dict
import pandas as pd

INPUT_FILE_NAME = 'output.txt'
OUTPUT_FILE_NAME = 'output_cleaned.txt'

def get_url(start=1):
    return 'https://search.naver.com/search.naver?&where=news&query=뉴스&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=10&start=%d' % start


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


news = []
start = 1
end = 4000
step = 10
bar = ProgressBar((end - start) // step, 20)
for i in range(start, end, step):
    pbar = bar.get_progress_bar((i - start) // step)
    if pbar is not None:
        print('[Link Crawling] %d / %d ' % (i, end) + pbar)

    url = get_url(i)
    req = requests.get(url)

    soup = BeautifulSoup(req.text, 'html.parser')
    a = soup.select('a._sp_each_url')
    for elem in a:
        link = elem['href']
        if link.startswith('https://news.naver.com'):
            news.append(elem['href'])

from PyKomoran import Komoran
k = Komoran('EXP')

def is_sentense(text):
    sep = k.get_list(text)
    for elem in sep:
        if elem.second == 'SW' and elem.second not in ['$%#@*&']:
            return False
        if elem.second == 'EF':
            return True
    return False

def title_is_hangul(text):
    return re.match(r'.*[ㄱ-ㅣ가-힣]+.*', text)

data = {'title': [], 'body': []}
bar = ProgressBar(len(news), 20)
for i, url in enumerate(news):
    pbar = bar.get_progress_bar(i)
    if pbar is not None:
        print('[Article Crawling] %d / %d ' % (i, len(news)) + pbar)

    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    selected = soup.select_one('h3#articleTitle')
    if selected is None:
        continue

    title = selected.get_text()
    title = re.sub(r'\[.+\] *', '', title)

    selected = soup.select('div#articleBodyContents > *')
    if selected is None:
        continue

    body = '\n'.join([text for text in [text.get_text() for text in selected] if is_sentense(text)])
    body = re.sub(r'\n{2,}', '\n', body)

    if body.strip() and title_is_hangul(title):
        data['title'].append(title)
        data['body'].append(body)

df = pd.DataFrame(data)

import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(df, f)
