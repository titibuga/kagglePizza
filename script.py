

import json
from random import shuffle
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation

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

shuffle(training_data)


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
vect = DictVectorizer(sparse=False)
X = vect.fit_transform(training_data)

#print X[0]
#print training_data[0]

N = len(X)

clf = RandomForestClassifier()
clf = clf.fit(X[N/5:], target_data[N/5:])



print clf.score(X[:N/5], target_data[:N/5])

print "===============\n"




