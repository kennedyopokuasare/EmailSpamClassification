# EmailSpamClassification

Implementation of Natural Language Processing and Text Mining course with the following specifications:

1. Study the examples in the blog https://appliedmachinelearning.wordpress.com/2017/01/23/email-spam-filter-python-scikit-learn/ it provides a good tutorial on every step in building the spam-filtering classifier. You need to check you program is working properly and be able to evaluate its performance (classification accuracy). You should evaluate your testing using Euron-spam corpus (see link in the blog document)

2. You may notice that the classifier is built around a feature set constituted of 3000 vocabulary words, recording the frequency of words in the document. Repeat the procedure using document-frequency times inverse document based feature. Compare the classification results.

3.You may notice that the program uses SVM and Naives’ Bayes classifiers. Try to repeat the testing using other classifiers that you may call upon from skit-learn package. Use the majority voting rule to aggregate the result of the various classifiers (e.g., counting the number of classifiers that yield spam and those that yield ham, and then select the class assigned by the largest number of classifiers). Discuss your results.

4.We would like to implement other feature set. Let the feature set be F={F1 F2 F3 F4 F5], where
F1 = number of URL, links in the message
F2 = number of language mistakes in the message
F3 = number of words in the text
F4 = number of named entities in the message
F5 = number of suspected links (classified as unsafe, or no matching with words before or after the link)
  -  Design your own strategy for implementing the above feature components
  -  Repeat the classification task using the new feature set and discuss the result with the previous case study.
  
5. We want to use another feature set extracted from TF.IDF in 2). First use the principal component analysis PCA available in sklearn-decomposition, and construct a feature vector of 10 highest PCA vector set from the feature set 2).  Repeat the training of classifiers  and the testing.

6. Repeat 5) using latent discriminant analysis dimension reduction (also available in sklearn python package. Use a 10 component dimension vector for the feature set, and repeat the training and classification of the classifiers.

7. Construct a new feature set constituted of a concatenation of PCA and LDA feature vectors. Repeat the training and testing of the classifiers.  Conclude about the performance of the classification with respect to various feature sets.

8. Suggest a new feature representation of your own that you may think is more relevant to the nature of message you have, and test its performances with the classifiers. 

