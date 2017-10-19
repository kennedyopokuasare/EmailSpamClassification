#### Dependencies
# 1. numpy
# 2. scipy
# 2. scikit-learn

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer


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
                    emailClass=mail.split(".")[-2]
                    all_email_corpus['text'].append(' '.join(email_words))
                    all_email_corpus['class'].append(emailClass)
                        
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
    np.save('all_email_corpus',all_email_corpus)
    
    return vocabulary,all_email_corpus
def evaluate_learning(tn, fp, fn, tp ):
   
    TP=tp
    FP=fp
    FN=fn
    TN=tn
   

    recall=float(TP)/(TP+FN)
    precision=float(TP)/(TP+FP)
    accuracy=float(TP+TN)/(TP+FP+FN+TN)
    error=float(FP+FN)/(TP+FP+FN+TN)
  
    return {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'precision':precision,'recall':recall,'accuracy':accuracy,'error':error}
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

def classify_SVM_NB(y_test,svn_prediction,nb_prediction):
    print "SVM"
    tn, fp, fn, tp= confusion_matrix(y_test, svn_prediction).ravel()
    print evaluate_learning(tn, fp, fn, tp )
    print "Naive Bayesian"
    tn, fp, fn, tp= confusion_matrix(y_test, nb_prediction).ravel()
    print evaluate_learning(tn, fp, fn, tp )

# ### creating the incidence matrix
# #### Create a dictionary of words with its frequency
root_dir = 'dataset'
make_Dictionary(root_dir)
#vocabulary,all_email_corpus = make_Dictionary(root_dir)
#print vocabulary

##### Prepare feature vectors per training mail and its labelsf# In[ ]:

#print dictionary
#print all_email_corpus

#dictionary=np.load("dict_enron.npy").tolist()
#print dictionary

#features_matrix,labels = extract_features(root_dir)

#np.save('enron_features_matrix.npy',features_matrix)
#np.save('enron_labels.npy',labels)

#countVectorizer=CountVectorizer()
#countVectorizer.vocabulary
#incident_matrix = np.load('enron_features_matrix.npy')
#labels = np.load('enron_labels.npy')

#tfIdf=compute_tf_idf(incident_matrix)

##transformer=DictVectorizer()
####features=[{"tf":incident_matrix},{"tf.idf":tfIdf}]
#featureSet=transformer.fit_transform(features)
#featureSet=transformer.transform(features)

#pipeline=pipeline[('tf',incident_matrix),('tf_idf',tfIdf),('classifier',LinearSVC())]

#print "Ground Truth Ham Spam"
#print sum(labels==0),sum(labels==1)

##X_train, X_test, y_train, y_test = train_test_split(featureSet, labels, test_size=0.40)


#### Training models and its variants


#model1 = LinearSVC()
#model2 = MultinomialNB()

#print  "training model using tf" 
#model1.fit(X_train,y_train)
#model2.fit(X_train,y_train)

#result1 = model1.predict(X_test)
#result2 = model2.predict(X_test)

#classify_SVM_NB(y_test,result1,result2)


#print  "training model using tf.idf"

#model1.fit(X_train,y_train)
#model2.fit(X_train,y_train)

#result1 = model1.predict(X_test)
#result2 = model2.predict(X_test)

#classify_SVM_NB(y_test,result1,result2)




