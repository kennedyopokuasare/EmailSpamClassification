#### Dependencies
# 1. numpy
# 2. scipy
# 2. scikit-learn

import os
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin

# #### Extracting F1 = number of URL, links in the message


    



# run this method onces, and then load the saved data and use subsequently
# method saves dict_enron.npy, all_email_corpus
def make_Dictionary(root_dir):
    all_email_corpus={'text': [], 'class': []}
   
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:                    
                    email_words=[]
                    for line in m:
                        words = line.split()
                        all_words += words
                        email_words+=words
                    emailClass='ham'
                    print mail.split(".")[-2]
                    if mail.split(".")[-2]=='spam':
                        emailClass='spam'
                    all_email_corpus['text'].append(' '.join(email_words))
                    all_email_corpus['class'].append(emailClass) #1 is spam , 0 is ham
                        
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    vocabulary=sorted( [key for (key,value) in dictionary]) 
    np.save('vocabulary.npy',vocabulary) 
    np.save('all_email_corpus.npy',all_email_corpus)
    
    return vocabulary,all_email_corpus

def evaluate_prediction(labels_test,predictions):
    evaluationTable=[]
    for key,value in predictions.iteritems():
        
        evaluation={}
        evaluation['Classifier']=key
        

        evaluation['Recall']=recall_score(labels_test,value)
        evaluation['Precision']=precision_score(labels_test,value)
        evaluation['F1 Score']=f1_score(labels_test,value)
        evaluation['Average Precision score']=average_precision_score(labels_test,value)
        
        evaluationTable.append(evaluation)
    return evaluationTable
class URLCountVectorizer(BaseEstimator,TransformerMixin):
    """Takes a list of documents and extracts count of URLs,links in document"""
    def __init__(self):
        pass
    def count_url_links(self, s):
        """Returns number of emails found in text"""
        express1="((?:(http s?|s?ftp):\/\/)?(?: www \.)?((?:(?:[A-Z0-9][A-Z0-9-]{0,61}[A-Z0-9]\.)+)([A-Z]{2,6})|(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))(?::(\d{1,5}))?(?:(\/\S+)*))"
        express2="http [s]?:// (?: www \.)? (?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        express3="http|www|goto"

        re1='(http|https|goto)'	# Word 1
        re2='(\\s+)'	# White Space 1
        re3='(.)'	# Any Single Character 1
        re4='(\\s+)'	# White Space 2
        re5='(.)'	# Any Single Character 2
        re6='(\\s+)'	# White Space 3
        re7='(\\/)'	# Any Single Character 3
        re8='.*?'	# Non-greedy match on filler
        re9='((www)*)'	# Word 2
        re10='(\\s+)'	# White Space 4
        regex = re.compile(re1+re2+re3+re4+re5+re6+re7+re8+re9+re10,re.IGNORECASE|re.DOTALL)
        
        emails=re.findall(regex, s)

        return len(emails)

    def get_all_url_counts(self, docs):
        """Encodes document to number of URL, links"""
        
        return [self.count_url_links(d) for d in docs]

    def transform(self, docs, y=None):
        """The workhorse of this feature extractor"""
        resultList=self.get_all_url_counts(docs)
        return np.transpose(np.matrix(resultList))

    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

#### Step 0. extracting email corpus and vocabulary
#root_dir = 'dataset'
#make_Dictionary(root_dir)

#### step 0.1. 
all_email_corpus=np.load("all_email_corpus.npy").item()
vocabularyList=np.load("vocabulary.npy").tolist()
documents=all_email_corpus['text']
labels=all_email_corpus['class']

binarizer=LabelBinarizer()
binarisedLables=binarizer.fit_transform(labels).ravel()

#x is document
# y is the labels or classes
document_train, document_test, labels_train, labels_test = train_test_split(documents, binarisedLables, test_size=0.40)

### step 1. Only term frequency feature
print "With TF"
word2vectTransformer=CountVectorizer(vocabulary=vocabularyList)

SVM_pipeline=Pipeline([
    ('tf',word2vectTransformer),
    ('SVM',LinearSVC()) 
])
NB_pipeline=Pipeline([
    ('tf',word2vectTransformer),
    ('SVM',MultinomialNB()) 
])
RF_pipeline=Pipeline([
    ('tf',word2vectTransformer),
    ('Random Forest',RandomForestClassifier()) 
])
KNN_pipeline=Pipeline([
    ('tf',word2vectTransformer),
    ('Random Forest',KNeighborsClassifier()) 
])



predictions={}
predictions["SVM"]=SVM_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['Naive Bayesian']=NB_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['Random Forest']=RF_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['K Nearest Neighbour']=KNN_pipeline.fit(document_train,labels_train).predict(document_test)
#print type(predictions)

scores=evaluate_prediction(labels_test,predictions)


print scores

#### Step 2.  TF.IDF
print "With TF.IDF"
documents2TfidfVector =TfidfVectorizer(vocabulary=vocabularyList)
SVM_pipeline=Pipeline([
    ('tfIdf',documents2TfidfVector),
    ('SVM',LinearSVC()) 
])
NB_pipeline=Pipeline([
    ('tfIdf',documents2TfidfVector),
    ('SVM',MultinomialNB()) 
])
RF_pipeline=Pipeline([
    ('tf',documents2TfidfVector),
    ('Random Forest',RandomForestClassifier()) 
])
KNN_pipeline=Pipeline([
    ('tf',documents2TfidfVector),
    ('Random Forest',KNeighborsClassifier()) 
])

predictions={}
predictions["SVM"]=SVM_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['Naive Bayesian']=NB_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['Random Forest']=RF_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['K Nearest Neighbour']=KNN_pipeline.fit(document_train,labels_train).predict(document_test)

scores=evaluate_prediction(labels_test,predictions)

print scores

#### Step 3. Feature component set F1...F5

print "using feature set F1.."

#document2NoURL=[ {'number_of_emails':count_url_links(d)} for d in document_train]

#print type(document2NoURL)
#print document2NoURL
document2URL_count_vector=URLCountVectorizer()
#document2URL_count_vector.vocabulary_=document2NoURL
#document2URL_count_vector.feature_names_=labels_train

featureSet=FeatureUnion([
    ('Count of URLs',document2URL_count_vector)
])
SVM_pipeline=Pipeline([
    ('feature set',featureSet),
    ('SVM',LinearSVC()) 
])
NB_pipeline=Pipeline([
    ('feature set',featureSet),
    ('SVM',MultinomialNB()) 
])
RF_pipeline=Pipeline([
    ('feature set',featureSet),
    ('Random Forest',RandomForestClassifier()) 
])
KNN_pipeline=Pipeline([
    ('feature set',featureSet),
    ('Random Forest',KNeighborsClassifier()) 
])

predictions={}

predictions["SVM"]=SVM_pipeline.fit( document_train,labels_train).predict(document_test)
predictions['Naive Bayesian']=NB_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['Random Forest']=RF_pipeline.fit(document_train,labels_train).predict(document_test)
predictions['K Nearest Neighbour']=KNN_pipeline.fit(document_train,labels_train).predict(document_test)

scores=evaluate_prediction(labels_test,predictions)

print scores