#### Dependencies
# 1. numpy
# 2. scipy
# 2. scikit-learn

import os
import numpy as np
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score

# #### Extracting F1 = number of URL, links in the message
def count_emails(s):
    """Returns an iterator of matched emails found in string s."""
    
    regex = re.compile(r"((?:(https?|s?ftp):\/\/)?(?:www\.)?((?:(?:[A-Z0-9][A-Z0-9-]{0,61}[A-Z0-9]\.)+)([A-Z]{2,6})|(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))(?::(\d{1,5}))?(?:(\/\S+)*))", re.IGNORECASE)

    emails=re. re.findall(regex, s)
    return len(emails)

def compute_tf_idf(incidentMatrix):
    "Returns the TFF.IDF of the Incident Matrix"
    transformer =TfidfTransformer(smooth_idf=False)
   
    transformer.fit(incidentMatrix)
    tfIdf=transformer.transform(incident_matrix)
    return tfIdf
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
def extract_features(root_dir): 
    
    docID = 0
    #features_matrix = np.zeros((33716,3000))
    features_matrix = np.zeros((200,3000))
    train_labels = np.zeros(200)
    # at this point we can load the saved all emails and dictionary
    #emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  
    all_email_corpus=np.load("all_email_corpus.npy").item()
    vocabulary=np.load("vocabulary.npy").tolist()
   
    #for mail in all_email_corpus:

  #  for emails_dir in emails_dirs:
  #      print "in email directory"
  #      dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
  #      for d in dirs:
  #          emails = [os.path.join(d,f) for f in os.listdir(d)]
  #          for mail in emails:
  #              print mail
  #              with open(mail) as m:
  #                  all_words = []
 ##                  for line in m:
  #                      print "iterating emails"
 ##                      words = line.split()
 #                       all_words += words
 #                   print all_words
                 
    docID=0
    for key, emails in all_email_corpus.iteritems():
        for email in emails:
            for word in email:                                
                print "extracting word Term Frequency of words"
                wordID = 0
                if (word.isalpha()) and (word in vocabulary):
                    print "word in vocabulary"
                    wordID = vocabulary.index(word)
                    wordTermFrequency=email.count(word)                                   
                    features_matrix[docID,wordID] = wordTermFrequency
            train_labels[docID] = int(key == 'spam')
            docID = docID + 1                
    return features_matrix,train_labels

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