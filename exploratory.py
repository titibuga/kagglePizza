import numpy as np
import json
from datetime import datetime
from nltk.stem import WordNetLemmatizer

## nltk
import nltk
from collections import defaultdict
#######
from nltk.corpus import stopwords

"""

Very inefficient code, jsut a quick aproach to have a idea
about the shape of the data


"""



with open('train.json', 'r') as f:
    training_data = json.load(f)

with open('test.json', 'r') as f:
    test_data = json.load(f)

usefulKeys = []
print "========= Useful keys: =============\n"
for k in training_data[1].keys():
    if k in test_data[1].keys():
        usefulKeys.append(k)
        print "'%s',"%( k)



#Useless keys...for the record
uselessKeys = [k for k in training_data[1].keys() if k not in usefulKeys]
print "========= Useless keys: =============\n"
for k in uselessKeys:
        print "'%s',"%( k)

targetKey = 'requester_received_pizza'
#############################################
########## Unix timestamp

key = 'unix_timestamp_of_request_utc'
timestampsUTC = [d[key] for d in training_data]
timestampsUTCSuc = [d[key] for d in training_data if d[targetKey]]
timestampsUTCFail = [d[key] for d in training_data if not d[targetKey]]
timestampsUTC = [datetime.fromtimestamp(bla) for bla in timestampsUTC]
timestampsUTCSuc = [datetime.fromtimestamp(bla) for bla in timestampsUTCSuc]
timestampsUTCFail = [datetime.fromtimestamp(bla) for bla in timestampsUTCFail]
yearReqs = np.array([date.year for date in timestampsUTC])
minYear = np.amin(yearReqs)

yearReqs = set(yearReqs)
yearSucs = {}
yearFails = {}
yearTotal = {}
for year in yearReqs:
    yearSucs[year] = yearFails[year] = yearTotal[year] = 0

for d in timestampsUTCSuc:
    yearSucs[d.year] += 1
    yearTotal[d.year] += 1

for d in timestampsUTCFail:
    yearFails[d.year] += 1
    yearTotal[d.year] += 1

print "\n============ Request date information ==============\n"
print "Earliest year post:%d"%(minYear)
print " Table of success and failure per year"
print "Year\tSuccesses\tFails"
for year in sorted(yearTotal.keys()):
    print "%d\t%d(%f)\t%d(%f)"%(year, yearSucs[year], yearSucs[year]/float(yearTotal[year]),
yearFails[year], yearFails[year]/float(yearTotal[year]))

print "\n"



#####################################

print "========== giver_username_if_known ========== "
naGUser = [d for d in training_data if d['giver_username_if_known'] == 'N/A']
naGUserSuc = [d for d in naGUser if d[targetKey]]
knownGUser = [d for d in training_data if d['giver_username_if_known'] != 'N/A']
knownGUserSuc = [d for d in knownGUser if d[targetKey]]

print "N/A giving users: %d | Success rate: %f "%(len(naGUser), len(naGUserSuc)/float(len(naGUser)))
print "Known giving users: %d | Success rate: %f "%(len(knownGUser), len(knownGUserSuc)/float(len(knownGUser)))




#######################

print "\n ============= REQUEST TEXT INFORMATION =========="
"""
for d in training_data:
    if len(d['request_text']) == 0
"""

text0 =len( [d for d in training_data if len(d['request_text']) == 0] )

print "Number of posts with empty body: %d from %d\n "%(text0, len(training_data))

successTextSize = np.array([len(d['request_text']) for d in training_data if d[targetKey] ==  True] ) 
losingTextSize = np.array([len(d['request_text']) for d in training_data if d[targetKey] ==  False] ) 


print "Mean of text length in case of:"
print "\t- Success:%f"% (successTextSize.mean()) 
print "\t- Failure:%f"% (losingTextSize.mean()) 

print"\n"

successTitleSize = np.array([len(d['request_title']) for d in training_data if d[targetKey] ==  True] ) 
losingTitleSize = np.array([len(d['request_title']) for d in training_data if d[targetKey] ==  False] ) 

print "Mean of tile length in case of:"
print "\t- Success:%f"% (successTitleSize.mean()) 
print "\t- Failure:%f"% (losingTitleSize.mean()) 

print"\n"

### Analyzinh words

stops = stopwords.words('english')
words = defaultdict(int)
successgivenword = defaultdict(float)
words_success = defaultdict(int)
words_fail = defaultdict(int)
lem = WordNetLemmatizer()

for d in training_data:
    t =  nltk.tokenize.word_tokenize( d['request_title']+' '+ d['request_text'])
    t = set([w.lower() for w in t])
    
    for w in t:
        w = w.lower()
        #w = lancaster_stemmer.stem(w)
        w = lem.lemmatize(w)
        
        
        if w.isalpha() and w not in stops:
            words[w] += 1
            if d["requester_received_pizza"]:
                words_success[w] += 1
            else:
                words_fail[w] += 1
for w in words:
	if words[w] > 10:
		successgivenword[w] = words_success[w]/float(words[w])

nWords = 20

print "Estimate of the probability P(success | have word). Print only the top %d"% (nWords)
print "Format: - word: prob (How many suc posts)" 

sortedWords = sorted(successgivenword, key=successgivenword.get, reverse=True)

for w in sortedWords[:nWords-1]:
    print "\t- %s: %f (%d)"%(w,successgivenword[w],words_success[w])
    


