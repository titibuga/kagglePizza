#with inspiration from http://hackersome.com/p/piskvorky/gensim/watchers

import logging, json
import itertools
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import gensim



def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def generator_testAndTitles(file):
    with open(file, 'r') as f:
        train_json = json.load(f)


    for request in train_json:
        title = request['request_id']
        tokens = gensim.utils.to_unicode(request['request_title'] + " " + request['request_text_edit_aware'], 'latin1').strip()
        tokens = tokenize(tokens)

        yield title, tokens


class Corpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(generator_testAndTitles(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs




stream = generator_testAndTitles('train.json')
doc_stream = (tokens for _, tokens in stream)
id2word_wiki = gensim.corpora.Dictionary(doc_stream)
id2word_wiki.filter_extremes(no_below=30, no_above=0.1)

wiki_corpus = Corpus('train.json', id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)

gensim.corpora.MmCorpus.serialize('./wiki_bow.mm', wiki_corpus)
mm_corpus = gensim.corpora.MmCorpus('./wiki_bow.mm')

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 8000)  # use fewer documents during training, LDA is slow
lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=50, id2word=id2word_wiki, passes=8)

_ = lda_model.print_topics(-1)  # print a few most important words for each LDA topic
tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)
gensim.corpora.MmCorpus.serialize('./data/wiki_tfidf.mm', tfidf_model[mm_corpus])
tfidf_corpus = gensim.corpora.MmCorpus('./data/wiki_tfidf.mm')
print(tfidf_corpus)


same_lda_model = gensim.models.LdaModel.load('./data/lda_wiki.model')

top_words = [[word for _, word in lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]
print(top_words)
lda_model.print_topics(-1)

