import io, json, nltk, pprint, sklearn
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.linear_model import LogisticRegression


word_features = {
"caus","lou","red","though","quit","annivers","mo","happy","delay","light","ask","zealand","boy","loan","missour","flat","aid","repay","lat","littl","peanut","univers","sunday","vancouv","goe","god","sant","subreddit","rough","grat","pic","holiday","bik","raop","lon","ont","amaz","dalla","northern","los","warm","mad","part","chees","pow","driv","terr","ago","dad","check","return","co","semest","john","fail","dog","moth","girl","il","forward","singl","tuesday","hold","cle","fil","embarrass","project","say","stamp","bor","prom","ok","humbl","loc","payday","coupl","pregn","mor","fav","comfort","friday","city","without","cut","around","austin","chicago","sint","wait","ther","hospit","almost","anyth","insid","thursday","cold","virgin","way","luck","away","cook","kid","pay","middl","fal","ten","sud","portland","told","med","stressed","minneapol","connecticut","cak","heard","chil","eng","writ","dream","somebody","wont","pantry","pul","wisconsin","honest","baltim","hug","ram","surv","texa","busy","art","debt","und","pb","catch","employ","broth","ind","ia","receiv","party","thought","savannah","hal","read","bit","hung","run","arkansa","homeless","info","problem","dying","phoenix","died","deserv","feel","rent","wif","mak","tonight","unemploy","hard","last","husband","fridg","car","sad","com","fiant","reddit","oth","tx","tre","sur","wil","year","night","md","ks","carolin","doll","poor","oregon","account","cat","know","din","nc","weekend","hop","rainy","exchang","provid","random","surgery","west","fund","near","rest","world","next","mi","gradu","spend","sou","get","meal","ev","hour","someth","monday","look","sick","four","alabam","vt","spent","second","search","smal","che","ms","unfortun","la","yo","feet","fast","miam","phil","bring","boss","dur","ir","antonio","choos","clos","couch","screwed","temp","puppy","ohio","help","hot","alon","wednesday","fl","ontario","someon","post","paycheck","surpr","also","domino","al","liv","bil","az","hungry","story","apprecy","starv","ga","nev","put","stil","wond","tot","bed","seek","pa","scotland","pack","dat","gift","song","money","colleg","tomorrow","try","fee","stat","stuck","ov","vet","food","paid","bad","tight","pleas","stud","lost","giv","sob","brok","delicy","famy","un","new","left","two","va","cent","desp","complet","son","lif","hav","job","go","first","work","real","old","cool","kansa","horr","sum","seattl","teach","nyc","lab","deposit","found","mind","kind","sev","struggling","week","everyth","yet","d","camp","request","day","californ","ny","us","lik","wa","nee","washington","bank","pap","tn","uk","canad","grad","would","could","act","whil","got","tim","buy","ca","sup","morn","start","cal","half","bright","mus","exam","sound","els","draw","aft","gf","gam","michig","celebr","par","lov","bef","pizz","atlant","worst","mn","favorit","minnesot","turn","pain","brooklyn","abl","oklahom","sust","instead","shit","eat","sist","empty","thing","long","som","guy","beach","noth","show","denv","jobless","pennsylvan","bc","low","amp","hom","ar","bet","end","fre","thi","today","gt","doe","young","past","pass","york","may","stress","sit","lt","trad","man","spar","plan","pretty","hungov","hit","wish","thanksg","wel","person","sydney","orlando","poss","til","nic","crav","lunch","becaus","indian","afford","ky","awesom","school","columb","send","depress","oh","anyon","visit","southern","short","think","san","wi","mom","hous","extrem","extr","cheesy","ran","lit","wo","kick","good","rec","town","birthday","georg","fir","stomach","toronto","pie","right","sav","three","going","childr","lack","rel","ottaw","ready","video","ver","credit","anoth","pizza","wrong","las","bean","sc","thes","eg","ri","ath","hook","tummy","seem","shop","stol","bas","wor","mont","besid","film","ankl","supply","shitty","coupon","tamp","ak","diet","jacksonvil","back","gre","break","austral","friend","fin","high","plac","girlfriend","card","leav","chang","boyfriend","mayb","colorado","hey","tir","ric","went","mon","keep","midterm","crust","dorm","county","glad","soup","meet","point","sandwich","law","philadelph","absolv","band","req","melbourn","bel","marathon","sacramento","bellingham","across","jersey","ct","anybody","deal","spring","larg","mov","peopl","pick","illino","stay","month","movy","main","rock","believ","dont","louisvil","boston","gen","country","massachuset","might","tast","chick","wallet","due","grocery","watch","want","room","much","decid","top","ston","bay","tough","cash","wer","op","study","thank","interview","england","hut","im","east","payp","fort","ut","enjoy","tak","best","teen","crazy","pittsburgh","min","find","ireland","ful","florid","diego","let","ma","saturday","miss","took","stuff","apart","fuck","gon","see","hel","london","houston","enough","play","slic","ord","noodl","yummy","accid","queen","budget","stressful","buck","chant","unit","company","incom","onlin","shar","lak","nashvil","replac","forgot","debit","yup","storm","park","class","shot","ment","fuel","soon","dough","bal","dump","louisian","adv","wv","drunk","dud","beer","many","cant","alask","glut","pepperon","mood","wan","wal","tasty","na","nat","test"
}

def request_simpleWords(request):

	lancaster_stemmer = LancasterStemmer()
	request_words = nltk.tokenize.word_tokenize(request)
	
	request_words = [w.lower() for w in request_words]
	request_words = [w for w in request_words if w.isalpha() ]
	request_words = [lancaster_stemmer.stem(w) for w in request_words]
	request_words = set(request_words)
	
	features = {}
	for word in word_features:
		features['contains({})'.format(word)] = (word in request_words)
	return features
	
	
with io.open('train.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)
	
featuresets = []
for d in training_data:
	featuresets.append(( request_simpleWords(d["request_text"]) ,  d["requester_received_pizza"] ))
	
#train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.classify.SklearnClassifier(LogisticRegression()).train(featuresets)
#classifier = nltk.classify.NaiveBayesClassifier.train(featuresets)
#print(nltk.classify.accuracy(classifier, test_set))
#print classifier.show_most_informative_features(25)


with open('test.json', 'r') as f:
    test_data = json.load(f)

f = open('testpredict-simpleWordsModel.csv', 'w')
f.write("request_id,requester_received_pizza\n");

for d in test_data:
	dprob = classifier.prob_classify(request_simpleWords(d["request_text_edit_aware"]))
	f.write("%s,%f\n" % (d["request_id"], dprob.prob(True)))

f.close()
	
	
	
