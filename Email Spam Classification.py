# coding=utf-8
# Dependencies
# 1. numpy
# 2. scipy
# 2. scikit-learn

import os
import numpy as np
import nltk
import re
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate,cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer,scale
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score,accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from nltk.tag.stanford import StanfordNERTagger
import grammar_check

class EmailClassifier():
    """
     Classifies emails into spam or ham based on several features 
     The Classifier uses the Support Vector Machine, Random Forest, Naive Bayesian, K Nearest Neighbour
     The classifier using cross validation in training and evaluation, and also implements a majority votinig rule
     to classify emails into spam or ham
    """
    def __init__(self):
        pass

    # run this method onces, and then load the saved data and use subsequently
    # method saves dict_enron.npy, all_email_corpus
    def make_Dictionary(root_dir):
        all_email_corpus = {'text': [], 'class': []}

        emails_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        all_words = []
        for emails_dir in emails_dirs:
            dirs = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]
            for d in dirs:
                emails = [os.path.join(d, f) for f in os.listdir(d)]
                for mail in emails:
                    with open(mail) as m:
                        email_words = []
                        for line in m:
                            words = nltk.word_tokenize(line)  # line.split()
                            all_words += words
                            email_words += words
                        emailClass = 'ham'
                        print mail.split(".")[-2]
                        if mail.split(".")[-2] == 'spam':
                            emailClass = 'spam'
                        all_email_corpus['text'].append(' '.join(email_words))
                        all_email_corpus['class'].append(
                            emailClass)  # 1 is spam , 0 is ham

        dictionary = Counter(all_words)
        list_to_remove = dictionary.keys()

        for item in list_to_remove:
            if item.isalpha() == False:
                del dictionary[item]
            elif len(item) == 1:
                del dictionary[item]
        dictionary = dictionary.most_common(3000)
        vocabulary = sorted([key for (key, value) in dictionary])
        np.save('vocabulary.npy', vocabulary)
        np.save('all_email_corpus.npy', all_email_corpus)

        return vocabulary, all_email_corpus

    def classify_emails(self,SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables):

            classifier_Labels=["SVM","Naive Bayesian","Random Forest","K Nearest Neighbour(K=5)","Majority Vote"]
            majorityVoteClassifier=VotingClassifier(estimators=[("SVM",SVM_pipeline),("Naive Bayesian",NB_pipeline),("Random Forest",RF_pipeline),("K-Nearest Neigbour",KNN_pipeline)],voting='hard')
            for clf, label in zip([SVM_pipeline, NB_pipeline, RF_pipeline,KNN_pipeline, majorityVoteClassifier], classifier_Labels):
                scoring = ['accuracy','precision', 'recall', 'f1']
                crossValidationResults = cross_validate(clf, documents, binarisedLables, cv=2, scoring=scoring,return_train_score=False)
                accuracy=crossValidationResults['test_accuracy']
                precision=crossValidationResults['test_precision']
                recall=crossValidationResults['test_recall']
                f1=crossValidationResults['test_f1']
                metrics = {}
                metrics["Classifier"] = label
                metrics["Recall"] = "%0.2f (+/- %0.2f)" % (recall.mean(),recall.std())
                metrics["Precision"] = "%0.2f (+/- %0.2f)" % (precision.mean(),precision.std())
                metrics["F1 Score"] = "%0.2f (+/- %0.2f)" % (f1.mean(),f1.std())
                metrics["Accuracy"] = "%0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std())

            

                y_pred = cross_val_predict(clf, documents, binarisedLables)
                allSpam=[index for index,value in enumerate(y_pred) if value==1]
                allHam=[index for index,value in enumerate(y_pred) if value==0]
                spamCount= len(allSpam)
                HamCount= len(allHam)
                metrics["Spam Count"] = spamCount
                metrics["Ham Count"] = HamCount
                print metrics

                if label=="Majority Vote":
                    sampledSpam=random.sample(allSpam,5)
                    print "\n\n10 randomly sample spam examples"
                    for entry in sampledSpam:
                        print documents[entry]+"\n"
                    
                    sampledHam=random.sample(allHam,5)
                    print "\n\n10 randomly sample ham examples"
                    for entry in sampledHam:
                        print documents[entry]+"\n"
                scoring=[]
                accuracy=[]
                recall=[]
                precision=[]
                f1=[]
                metrics={}
                crossValidationResults={}
                allHam=[]
                allSpam=[]
                
    
    def evaluate_prediction(self,labels_test, predictions):
        evaluationTable = []
        for key, value in predictions.iteritems():
            confusion_matrix
            evaluation = {}
            evaluation["Classifier"] = key

            evaluation["Recall"] = recall_score(labels_test, value)
            evaluation["Precision"] = precision_score(labels_test, value)
            evaluation["F1 Score"] = f1_score(labels_test, value)
            evaluation["Accuracy"]=accuracy_score(labels_test,value)
            #evaluation["Average Precision score"] = average_precision_score(labels_test, value)
            evaluation["tn"],evaluation["fp"],evaluation["fn"],evaluation["tp"] = confusion_matrix(labels_test, value).ravel()
            evaluationTable.append(evaluation)
        return evaluationTable
    def using_TF_IDF_tutorial(self):
        print "With TF, IDF in tutorial"
        documents2TfidfVector =TfidfVectorizer(vocabulary=vocabularyList,decode_error='ignore')
        SVM_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('BN',MultinomialNB()) 
        ])
        
        document_train, document_test, labels_train, labels_test = train_test_split(documents, binarisedLables, test_size=0.40)
        predictions={}
        predictions["SVM"]=SVM_pipeline.fit(document_train,labels_train).predict(document_test)
        predictions['Naive Bayesian']=NB_pipeline.fit(document_train,labels_train).predict(document_test)
        
        #predictions['Random Forest']=RF_pipeline.fit(document_train,labels_train).predict(document_test)
        #predictions['K Nearest Neighbour']=KNN_pipeline.fit(document_train,labels_train).predict(document_test)
        # print type(predictions)

        scores=self.evaluate_prediction(labels_test,predictions)
        print scores
        #self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_TF_tutorial(self):
        print "With TF in tutorial"
        word2vectTransformer=CountVectorizer(vocabulary=vocabularyList,decode_error='ignore')

        SVM_pipeline=Pipeline([
            ('tf',word2vectTransformer),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('tf',word2vectTransformer),
            ('SVM',MultinomialNB()) 
        ])
        
        document_train, document_test, labels_train, labels_test = train_test_split(documents, binarisedLables, test_size=0.40)
        predictions={}
        predictions["SVM"]=SVM_pipeline.fit(document_train,labels_train).predict(document_test)
        predictions['Naive Bayesian']=NB_pipeline.fit(document_train,labels_train).predict(document_test)
        #predictions['Random Forest']=RF_pipeline.fit(document_train,labels_train).predict(document_test)
       #predictions['K Nearest Neighbour']=KNN_pipeline.fit(document_train,labels_train).predict(document_test)
        # print type(predictions)

        scores=self.evaluate_prediction(labels_test,predictions)
        print scores
        #self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_TF(self):
        """Uses term frequency feature for classification"""
        print "With TF"
        word2vectTransformer=CountVectorizer(vocabulary=vocabularyList,decode_error='ignore')

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
        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_TFIDF(self):
        """Uses term frequency x Inverse document frequency feature for classification"""
        print "With TF.IDF"

        java_path = "C:/Program Files/Java/jdk-9.0.1/bin/java.exe"
        os.environ['JAVAHOME'] = java_path
        nltk.internals.config_java(java_path)

        documents2TfidfVector =TfidfVectorizer(vocabulary=vocabularyList,decode_error='ignore')
        SVM_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('SVM',MultinomialNB()) 
        ])
        RF_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('Random Forest',RandomForestClassifier()) 
        ])
        KNN_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('Random Forest',KNeighborsClassifier()) 
        ])
        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_F1_F2_F3_F4_F5(self):
        """Uses the following features for classification,
            - Counts of Urls
            - Count of Language Mistakes
            - Count of words
            - Count of named entities
        """

        print "With feature set F1.F2.F3.F4"

        #Note LanguateMistakesVectorizer requires a running grammar-check server. check readme.md
        featureSet=FeatureUnion([
            ('Count of URLs',URLCountVectorizer()),
            #('Count of Language Mistakes',LanguageMistakesVectorizer()),
            ('Count of words',WordCountVectorizer()),
            ('Count of Named Entities',NameEntityCountVectorizer())
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
        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_PCA_of_TFIDF(self):
        """Uses 10 Principal Components of term frequency x Inverse document frequency feature for classification"""

        print "With PCA (TF.IDF)"
        documents2TfidfVector =TfidfVectorizer(vocabulary=vocabularyList,decode_error='ignore')
        SVM_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('PCA',PCA(n_components=10)),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('PCA',PCA( n_components=10)),
            ('Non Neg Scalling',PCAScaleTranformer()),
            ('SVM',MultinomialNB()) 
        ])

        RF_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('PCA',PCA( n_components=10)),
            ('Random Forest',RandomForestClassifier()) 
        ])
        KNN_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('PCA',PCA(n_components=10)),
            ('Random Forest',KNeighborsClassifier()) 
        ])
        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_LDA_of_TFIDF(self):
        """Uses linear discriminant analyses of term frequency x Inverse document 
        frequency feature for classification
        """

        print "Using LDA (TF.IDF)"
        documents2TfidfVector =TfidfVectorizer(vocabulary=vocabularyList,decode_error='ignore')
        SVM_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('LDA',LinearDiscriminantAnalysis(n_components=10)),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('LDA',LinearDiscriminantAnalysis( n_components=10)),
            ('Non Neg Scalling',PCAScaleTranformer()),
            ('SVM',MultinomialNB()) 
        ])

        RF_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('LDA',LinearDiscriminantAnalysis( n_components=10)),
            ('Random Forest',RandomForestClassifier()) 
        ])
        KNN_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('LDA',LinearDiscriminantAnalysis(n_components=10)),
            ('Random Forest',KNeighborsClassifier()) 
        ])

        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_PCA_and_LDA_of_TFIDF(self):
        """Uses 10 Principal Componets and Linear discriminant analyses of
         term frequency x Inverse document frequency feature for classification
         """
        documents2TfidfVector =TfidfVectorizer(vocabulary=vocabularyList,decode_error='ignore')
        print "Using PCA (TF.IDF), LDA (TF.IDF)"
        
        featureSet=FeatureUnion([
            ('PCA (TF.IDF)',PCA(n_components=10)),
            ('LDA',LinearDiscriminantAnalysis( n_components=10))
        ])

        SVM_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('featureset',featureSet),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('featureset',featureSet),
            ('Non Neg Scalling',PCAScaleTranformer()),
            ('SVM',MultinomialNB()) 
        ])

        RF_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('featureset',featureSet),
            ('Random Forest',RandomForestClassifier()) 
        ])
        KNN_pipeline=Pipeline([
            ('tfIdf',documents2TfidfVector),
            ('to_dense', DenseTransformer()), 
            ('featureset',featureSet),
            ('Random Forest',KNeighborsClassifier()) 
        ])

        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
    def using_suggested_features(self):
        """Uses targetted word list occurrence counts feature for classification"""

        print "Using target workds and symbols €,£,$,%,! viagra, penis, billion, billionaire, lottery, prize, charity , USA, Nigeria"

        SVM_pipeline=Pipeline([
            ('targeted_words',TargetedWordCountVectorizer()),
            ('SVM',LinearSVC()) 
        ])
        NB_pipeline=Pipeline([
            ('targeted_words',TargetedWordCountVectorizer()),
            ('SVM',MultinomialNB()) 
        ])
        RF_pipeline=Pipeline([
            ('targeted_words',TargetedWordCountVectorizer()),
            ('Random Forest',RandomForestClassifier()) 
        ])
        KNN_pipeline=Pipeline([
            ('targeted_words',TargetedWordCountVectorizer()),
            ('Random Forest',KNeighborsClassifier()) 
        ])

        self.classify_emails(SVM_pipeline,NB_pipeline,RF_pipeline,KNN_pipeline,documents,binarisedLables)
class WordCountVectorizer(BaseEstimator, TransformerMixin):
    """Takes a list of documents and extracts word counts in the document"""

    def __init__(self):
        pass

    def count_words_doc(self, doc):
        """Returns the count of words in a document"""
        all_words = doc.split()#nltk.word_tokenize(doc)
        all_words_iter = all_words
        for item in all_words_iter:
            if item.strip().isalpha() == False:
                all_words.remove(item)

        return len(all_words)

    def get_all_word_counts(self, docs):
         """Encodes document to number of words"""
         return [self.count_words_doc(d) for d in docs]

    def transform(self, docs, y=None):
        """The workhorse of this feature extractor"""
        resultList = self.get_all_word_counts(docs)
        return np.transpose(np.matrix(resultList))

    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
class NameEntityCountVectorizer(BaseEstimator, TransformerMixin):
    """Takes a document and extracts count of named entities.
        This class uses the NLTK parts of speach tagger
    """

    def __init__(self):
        pass

    def count_named_entities(self, doc):
        """Returns a count of named entities in te a document"""
        tokens = doc.split()#nltk.word_tokenize(doc)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        #serFile=os.path.join(dir_path,'english.all.3class.distsim.crf.ser')
        #jarFile=os.path.join(dir_path,'stanford-ner.jar')
        #st=StanfordNERTagger(serFile,jarFile,encoding='utf-8')
        #tokens_NER=tokens
        #for tk in tokens:
        #    if len(tk)==1:
        #        tokens_NER.remove(tk)
        #word_tag=st.tag(tokens_NER)
    
        named_entities_tag =[]
        #for tag in word_tag:
        #    if tag[1]!='O':
        #        named_entities_tag.append(tag)
    
        

        pos = nltk.pos_tag(tokens)
        
        ##NN	Noun, singular or mass
        ##NNS	Noun, plural
        ##NNP	Proper noun, singular
        ##NNPS	Proper noun, plural
        #namedEntityTags_set1 = [ "NN","NNS" ]#"NNPS""NN","NNS""NNP","NNPS"
        namedEntityTags_set2 = [ "NNP" ]#"NNPS""NNPS"
        named_entities = []
        named_entities_tag =[]
       
        for word, tag in pos:
            if tag in namedEntityTags_set2:
                if word.isalpha() and len(word)>1:
                   named_entities_tag.append(tag)
                   named_entities.append(word)
       # print named_entities

    
        return len(named_entities_tag)

    def get_all_named_entities(self, docs):
        """Encodes document to number of named entities"""
        return [self.count_named_entities(d) for d in docs]

    def transform(self, docs, y=None):
        """The workhorse of this feature extractor"""
        resultList=self.get_all_named_entities(docs)
       
        return np.transpose(np.matrix(resultList))

    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
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
class TargetedWordCountVectorizer(BaseEstimator,TransformerMixin):
    """Takes a list of documents and extracts the count of targeted words in the each document"""
    def __init__(self):
        pass
    def count_targeted_words(self,doc):
        target_words=["£","$","%","!" "viagra", "penis", "billion", "billionaire", "lottery", "prize", "charity" , "USA", "Nigeria"]
        target_hit_count= len([ word for word in target_words if word in doc ])
        return target_hit_count
        
    def get_all_targeted_words_count(self,docs):
        return [self.count_targeted_words(d) for d in docs]
        
    def transform(self, docs, y=None):
        """The workhorse of this feature extractor"""
        resultList=self.get_all_targeted_words_count(docs)
       
        return np.transpose(np.matrix(resultList))

    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
class LanguageMistakesVectorizer(BaseEstimator,TransformerMixin):
    """Takes a list of documents and extracts count of English language Mistakes"""
    _defaultLanguage='en-GB'
    def __init__(self,language=None):
        if language:
           self._defaultLanguage=language
        pass
    def count_language_mistakes(self, doc):
        """Returnes the count of English mistakes in the text """
        tool=grammar_check.LanguageTool(self._defaultLanguage)
        encodedText=doc.decode("utf-8",errors='replace')
        try:
            mistakes=tool.check(encodedText)
            
        except Exception  as e:
            mistakes=[]
        
        
        return len(mistakes)
    def get_all_mistake_count(self,docs):
         """Encodes document to number of Language mistakes"""
         return [self.count_language_mistakes(d) for d in docs]

    def transform(self, docs, y=None):
        """The workhorse of this feature extractor"""
        resultList=self.get_all_mistake_count(docs)
       
        return np.transpose(np.matrix(resultList))

    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
class DenseTransformer(TransformerMixin):
    "Takes a sparse array and converts it to dense array"

    def transform(self, X, y=None, **fit_params):
    
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
class PCAScaleTranformer(TransformerMixin):
    "Takes PCA Matrix and coverts to entries of absolute values"
    def transform(self, X, y=None, **fit_params):
       
        return np.absolute(X)
        
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

all_email_corpus=np.load("all_email_corpus.npy").item()
vocabularyList=np.load("vocabulary.npy").tolist()
documents=all_email_corpus['text']
labels=all_email_corpus['class']


binarizer=LabelBinarizer()
binarisedLables=binarizer.fit_transform(labels).ravel()

#document_train, document_test, labels_train, labels_test = train_test_split(documents, binarisedLables, test_size=0.40)

emailclassifier= EmailClassifier()
emailclassifier.using_TF_IDF_tutorial()