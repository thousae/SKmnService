import pickle

with open('crawling/news.pickle', 'rb') as f:
	data = pickle.load(f)

with open('crawling/news.pickle', 'wb') as f:
	pickle.dump(data, f, protocol=2)