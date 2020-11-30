import re
import pickle
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyKomoran import Komoran
from alive_progress import alive_bar


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def get_url(start=1):
    return 'https://search.naver.com/search.naver?&where=news&query=뉴스&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=10&start=%d' % start


def get_html_soup(url):
    req = requests.get(url)
    return BeautifulSoup(req.text, 'html.parser')


@static_vars(k=Komoran('EXP'))
def is_sentence(text):
    sep = is_sentence.k.get_list(text)
    for elem in sep:
        if elem.second == 'SW' and elem.second not in ['$%#@*&']:
            return False
        if elem.second == 'EF':
            return True
    return False


def is_hangul(text):
    return re.match(r'.*[ㄱ-ㅣ가-힣]+.*', text) is not None


def crawl_news_links(amount):
    news_links = []
    r = range(1, amount, 10)
    with alive_bar(amount) as bar:
        for i in r:
            soup = get_html_soup(get_url(i))
            a = soup.select('a._sp_each_url')
            for elem in a:
                link = elem['href']
                if link.startswith('https://news.naver.com'):
                    news_links.append(link)
            for _ in range(10):
                bar()
    print('%d news found!' % len(news_links))
    return news_links


def crawl_news_title_and_article(url):
    soup = get_html_soup(url)

    selected = soup.select_one('h3#articleTitle')
    if selected is None:
        return

    title = selected.get_text()
    title = re.sub(r'\[.+\] *', '', title)

    selected = soup.select('div#articleBodyContents > *')
    if selected is None:
        return

    body = '\n'.join([text for text in [text.get_text() for text in selected] if is_sentence(text)])
    body = re.sub(r'\n{2,}', '\n', body)

    if body.strip() and is_hangul(title):
        return title, body


def save_data(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def crawl(amount):
    news_links = crawl_news_links(amount)

    data = {'title': [], 'body': []}
    with alive_bar(len(news_links)) as bar:
        for url in news_links:
            result = crawl_news_title_and_article(url)
            if result is not None:
                title, body = result
                data['title'].append(title)
                data['body'].append(body)
            bar()

    print(str(len(data['title'])) + ' news crawled!')
    df = pd.DataFrame(data)
    save_data(df, 'crawling/news.pickle')


if __name__ == '__main__':
    crawl(4000)
