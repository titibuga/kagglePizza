import io
import json
from pprint import pprint
import sys
import time
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


stops = stopwords.words('english')

with io.open('train.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)
	
	
words = defaultdict(int)
words_success = defaultdict(int)
words_fail = defaultdict(int)
successgivenword = defaultdict(float)
lancaster_stemmer = LancasterStemmer()


for	d in training_data:
	t = nltk.tokenize.word_tokenize(d["request_title"])
	
	for w in t:
		w = w.lower()
		w = lancaster_stemmer.stem(w)
		
		if w.isalpha() and w not in stops:
			words[w] += 1
			if d["requester_received_pizza"]:
				words_success[w] += 1
			else:
				words_fail[w] += 1
		

for w in words:
	if words[w] > 5:
		successgivenword[w] = words_success[w]/float(words[w])

ff = open('simpleWords', 'w')
		
for w in sorted(successgivenword, key=successgivenword.get, reverse=True):
	if successgivenword[w] < 0.45 or successgivenword[w] > 0.55:
		print("\"%s\": %f," % (w, successgivenword[w]))
		ff.write("\"%s\"," % (w))
		
ff.close()

