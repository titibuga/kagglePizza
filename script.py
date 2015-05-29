

import json
from random import shuffle
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation, linear_model
from sklearn import metrics
import numpy as np
from ensembleC import EnsembleClassifier


def myCrossValid(model, X, y, folds = 5):
    aucs = []
    skf = cross_validation.ShuffleSplit(len(y), n_iter=folds,test_size = 0.25, random_state = 0)
    for train_index, test_index in skf:
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
#neigh = KNeighborsClassifier(n_neighbors=5)
#svc = SVC(probability = True)



#target_data2 = np.array([1 if t else 0 for t in target_data])
#scores = cross_validation.cross_val_score(clf, X, target_data2, scoring="roc_auc" , cv=10)
clf = clf.fit(X[N/5:], target_data[N/5:])
#neigh = neigh.fit(X[N/5:], target_data[N/5:])
#svc = svc.fit(X[N/5:], target_data[N/5:])


probs = clf.predict_proba(X[:N/5])
probs = clf.predict_proba(X[:N/5])
probs = [ p[1] for p in probs]
probs = np.array(probs)

target_data2 = [1 if t else 0 for t in target_data]
target_data2 = np.array(target_data2)

npizzas = 0
for t in target_data2:
    npizzas = npizzas + t

print ("Pizzas:{} / {}\n").format(npizzas, len(target_data))



#print metrics.roc_auc_score(target_data2, probs)


clf = EnsembleClassifier()
#model = linear_model.LogisticRegression(C=0.1, penalty='l1')

################ K-FOLD CROSS-VALID ###################


#aucs = myCrossValid(model,X,target_data2)

#################################################

#print "Cross-validation Accuracy:\n"
#print np.array(aucs).mean()



#### Temporary exit ####
#Just to stop the execution at this point...I don't want it
#to execute the rest of the code everytime (I should put these
#things in functions...)
#exit()
######################



print vect.get_feature_names()

clf.fit(X,target_data2)

# Let's now predict the test dataset 

with open('test.json', 'r') as f:
    test_data = json.load(f)


#print test_data[0].keys()
test_ids = [t["request_id"] for t in test_data]
#Remove ignored keys
for k in ignoredKeys:
    for d in test_data:
        if k in d:
            del d[k]


#Tranform the dataset in the format in which the tree was
# trained and make the prediction

X = vect.transform(test_data)
testprobs = clf.predict_proba(X)






#exit()
f = open('testpredict.csv', 'w')

f.write("request_id,requester_received_pizza\n");
for i in range(len(testprobs)):
    f.write(("{},{}\n").format(test_ids[i],testprobs[i][1]))

f.close()

print "THE END!!\n"
