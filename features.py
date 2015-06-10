import normalize, nltk, math
from datetime import datetime
import random as rand
from nltk.stem.lancaster import LancasterStemmer




def get_samples(training_data):
	samples = []  # n-array of (m-array of features) for each request
	targets = []  # n-array of outcome

	for d in training_data:
		features = []

		#features.append(got_pizza(d))
		#features.append(zero(d))
		#features.append(random(d))
		#features.append(title_length(d))
		features.append(post_length(d))
		#features.append(excitement_score(d))
		#features.append(count_http(d))
		features.append(count_imgur(d))
		#features.append(has_http(d))
		#features.append(has_kids(d))
		#features.append(has_pay_it_forward(d))
		#features.append(leet_score(d))
		#features.append(swear_score(d))
		#features.append(time(d))


		samples.append(features)
		targets.append(d["requester_received_pizza"])

	return samples, targets



def got_pizza(request):
	return request["requester_received_pizza"]

def zero(request):
	return 0

def random(request):
	return rand.uniform(0.0,1)

def title_length(request):
	text = request['request_title']
	return len(text)

def post_length(request):
	text = request['request_text_edit_aware']
	return len(text)

def excitement_score(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	return math.log(1 + text.count("!"), 2)

def count_http(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	return text.count("http")

def count_imgur(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	return text.count("imgur")

def has_http(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	if "http" in text:
		return 1
	return 0

def has_kids(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	vocab = normalize.vocab(text)

	words = ["kid", "toddl", "child", "childr", "son", "daught", "baby"]

	if any(word in words for word in vocab):
		return 1
	return 0

def has_pay_it_forward(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	text = text.lower()
	if "pay it forward" in text:
		return 1
	return 0

def leet_score(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	vocab = normalize.vocab(text)

	words = ["ty", "r", "u", "ru", "lol", "omg", "ily", "lmao", "wtf", "ppl", "idk", "tbh", "btw", "thx", "smh", "ffs", "ama", "fml", "tbt", "jk", "imo", "yolo", "rofl", "mcm", "ikr", "fyi", "brb", "gg", "idc", "tgif", "nsfw", "icymi", "stfu", "wcw", "irl", "bff", "ootd", "ftw", "txt", "hmu", "hbd", "tmi", "nm", "gtfo", "nvm", "dgaf", "fbf", "dtf", "fomo", "smfh", "omw", "potd", "lms", "gtg", "roflmao", "ttyl", "afaik", "lmk", "ptfo", "sfw", "hmb", "ttys", "fbo", "ttyn"]

	count = 0
	for w in words:
		if w in vocab:
			count += 1

	return count

def swear_score(request):
	text = request['request_title'] + " " + request['request_text_edit_aware']
	words = nltk.tokenize.word_tokenize(text)

	stemmer = LancasterStemmer()

	count = 0
	for w in words:
		w = w.lower()
		w = stemmer.stem(w)
		if w in ["fuck", "shit", "shitty"]:
			count += 1

	return count

def time(request):
	date = datetime.fromtimestamp(request['unix_timestamp_of_request_utc'])
	return date.year - 2011
	return request["unix_timestamp_of_request_utc"]



