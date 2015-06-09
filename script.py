

import json
from random import shuffle
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import random
from sklearn.feature_selection import RFE


#### TEXT ####
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


#####

from myFeatureExtractors import *


from sklearn import cross_validation, linear_model
from sklearn import metrics
import numpy as np
from ensembleC import EnsembleClassifier


def myCrossValid(model, X, y, folds = 5):
    aucs = []
    skf = cross_validation.ShuffleSplit(len(y), n_iter=folds,test_size = 0.25, random_state = 0)
    for train_index, test_index in skf:
        print "Cross-validating"
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        probs = model.fit(X_train, y_train).predict_proba(X_test)
        probs = [ p[1] for p in probs]
        aucs.append(metrics.roc_auc_score(y_test, probs))
    return aucs


    






targetKey = "requester_received_pizza"

#Useless or ~problematic~ (potentially null) or text
# features ignored for this first benchmark
ignoredKeys = [targetKey,
               "request_title",
               "request_text",
               "request_text_edit_aware",
               "request_id",
               "requester_username",
               "requester_subreddits_at_request",
               "requester_user_flair"]


with open('train.json', 'r') as f:
    training_data = json.load(f)



#Name of the features, removing the ignored ones
# PS: Not using it anymore, it's here just for the
# record
featureKeys =  training_data[0].keys()
for w in ignoredKeys:
    featureKeys.remove(w)

#shuffle(training_data)
target_data = [x[targetKey] for x in training_data]

#Add length of the message to the data set 
"""
for d in training_data:
    d['length_text'] = len( d['request_text'] )
    print  d['length_text']
"""


"""

#Remove from training data the ignored 
target_data = [x[targetKey] for x in training_data]
for k in ignoredKeys:
    for d in training_data:
        del d[k] 
#for item in training_data:
#    pItem = []
#    for k in featureKeys:
#    pItem.append(item[k])
#    training_data_parsed.append(pItem)

#Festures need to be numeric...sklearn
# provide this function...not very efficient.
vect = DictVectorizer(sparse = False)
X = vect.fit_transform(training_data)

#print X[0]
#print training_data[0]

N = len(X)

"""

target_data2 = [1 if t else 0 for t in target_data]
target_data2 = np.array(target_data2)

"""
select = SelectPercentile(score_func=chi2, percentile=10)
vect = TfidfVectorizer(ngram_range=(1, 5), analyzer="char")
messagesX = vect.fit_transform(messages)
messagesX = select.fit_transform(messagesX, target_data2)
"""

messageParser = Pipeline([
    ('messagesExtr', MessagesExtractor()),
    #('vect', CountVectorizer()),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 5), analyzer="char", stop_words='english')),
    ('select', LinearSVC(C=1, penalty="l1", dual=False)),
    ('dim_red', TruncatedSVD(n_components=100, random_state=42)),
])

basicFeatureParser = Pipeline([
    ('basicFeatExtr', BasicFeatureExtractor()),
    ('vect', DictVectorizer(sparse = False)),
    #('norm', Normalizer()),
   
])

ftUnion = FeatureUnion(
    transformer_list=[
        ('basic_ft', basicFeatureParser),
        ('message_ft', messageParser),
    ]
)

ftu = ftUnion.fit(training_data, target_data2)
X = ftu.transform(training_data) 




################### MODELS #####################

clf = RandomForestClassifier(n_estimators = 50)
#neigh = KNeighborsClassifier(n_neighbors=5)
#svc = SVC(probability = True)
## Naive bayes###

#### Naive Bayes

#naiveBayes = MultinomialNB()

ensemble = EnsembleClassifier()
logr = LogisticRegression(tol=1e-8, penalty='l1', C=1)




#############################################



#target_data2 = np.array([1 if t else 0 for t in target_data])
#scores = cross_validation.cross_val_score(clf, X, target_data2, scoring="roc_auc" , cv=10)
"""
clf = clf.fit(X[N/5:], target_data[N/5:])
#neigh = neigh.fit(X[N/5:], target_data[N/5:])
#svc = svc.fit(X[N/5:], target_data[N/5:])


probs = clf.predict_proba(X[:N/5])
probs = clf.predict_proba(X[:N/5])
probs = [ p[1] for p in probs]
probs = np.array(probs)
"""
#################

"""

### Calculating feature importance #####
#
#clf = ExtraTreesClassifier(n_estimators = 50)
vect = DictVectorizer(sparse = False)
basicTransf = BasicFeatureExtractor()
basicData = basicTransf.transform(training_data)
for b in basicData:
    #del b['requester_upvotes_minus_downvotes_at_request']
    b['random_n'] = random.uniform(0.0, 1.0)

print basicData[0].keys()

X = vect.fit_transform(basicData)
print vect.get_feature_names()
ftnames = [
        'random',
    'req_len',
    'acc_age',
    'days_fpost',
    'n_cmts',
    'n_cmts_raop',
    'n_posts',
    'n_post_raop',
    'n_subreddits',
    'up-d_votes',
    'up+d_votes',
    'timestamp',

]
estimator = SVC(kernel="linear")
print "hi"
selector = RFE(estimator, n_features_to_select=1, step=1)
print "hi2"
selector = selector.fit(X, target_data2)
print "hi2"
#clf = clf.fit(X, target_data2)
#ftimport = clf.feature_importances_
ftimport = selector.ranking_


for i in range(len(ftnames)):
    print "(%d):%s: %f"%(i+1,ftnames[i],ftimport[i])

bar_width = 0.35
index = np.arange(len(ftimport))


rects1 = plt.bar(index, ftimport, bar_width)

plt.xticks(index + bar_width/2,ftnames)
plt.tight_layout()
plt.show()




exit()


###################################
"""



#print metrics.roc_auc_score(target_data2, probs)



#model = linear_model.LogisticRegression(C=0.1, penalty='l1')

################ K-FOLD CROSS-VALID ###################


aucs = myCrossValid(ensemble,X,target_data2)
#aucs = cross_validation.cross_val_score(logr, 
               #          X, target_data2, cv=5, n_jobs=4, scoring='roc_auc')

#################################################

print "Cross-validation Accuracy:\n"
print np.array(aucs).mean()



#### Temporary exit ####
#Just to stop the execution at this point...I don't want it
#to execute the rest of the code everytime (I should put these
#things in functions...)
exit()
######################


clf = ensemble


#print vect.get_feature_names()

clf.fit(X,target_data2)

# Let's now predict the test dataset 

with open('test.json', 'r') as f:
    test_data = json.load(f)


#print test_data[0].keys()
k1 = 'request_text'
k2 = 'request_text_edit_aware'
test_ids = [t["request_id"] for t in test_data]
#for d in test_data:
#    if k1 in d.keys():
#        d['length_text'] = len( d[k1] )
#    else:
#        d['length_text'] = len( d[k2] )


#Remove ignored keys
"""
for k in ignoredKeys:
    for d in test_data:
        if k in d:
            del d[k]
"""


#Tranform the dataset in the format in which the tree was
# trained and make the prediction
print "==="
X = ftu.transform(test_data)
testprobs = clf.predict_proba(X)






#exit()
f = open('testpredict.csv', 'w')

f.write("request_id,requester_received_pizza\n");
for i in range(len(testprobs)):
    f.write(("{},{}\n").format(test_ids[i],testprobs[i][1]))

f.close()

print "THE END!!\n"
