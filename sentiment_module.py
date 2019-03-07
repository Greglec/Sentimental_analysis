#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import nltk
#nltk.download('averaged_perceptron_tagger')
import random
from nltk.classify.scikitlearn import SklearnClassifier #for scilearn classifier
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC,NuSVC

from nltk.classify import ClassifierI #so we can inherate from the nltk classifier class
from statistics import mode #for the classifier vote system


# In[2]:


class ScoreClassifier(ClassifierI): #we pass a list of classifiers through this class
    def __init__(self, *classifiers):#init method to run any methods
        self._classifiers = classifiers #classifier list will be whatever list of classifiers passed 
        
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)#returns number of votes
    
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[3]:


#saving documents and all_words in pickle
documents_f=open("documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("word_features.pickle","rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


# In[4]:


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features


# In[5]:


#convert words in dictionnary of 5000 words with category true (neg) or false (pos)
#featuresets = [(find_features(rev), category) for (rev,category) in documents ]

featuresets_f = open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()


# In[6]:


random.shuffle(featuresets)


# In[7]:


D_train = featuresets[:10000]
D_test = featuresets[10000:]


# In[8]:


open_file = open("NB_classifier.pickle","rb")
NB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("MNB_classifier.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("StochasticGradient_classifier.pickle","rb")
StochasticGradient_classifier = pickle.load(open_file)
open_file.close()

open_file = open("SVC_classifier.pickle","rb")
SVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

#open_file = open("voted_classifier.pickle","rb")
#voted_classifier = pickle.load(open_file)
#open_file.close()


# In[9]:


voted_classifier = ScoreClassifier(NB_classifier, MNB_classifier,BernoulliNB_classifier, LogisticRegression_classifier,StochasticGradient_classifier)

#save_classifier=open("voted_classifier.pickle","wb")
#pickle.dump(voted_classifier,save_classifier)
#save_classifier.close()

print("voted accuracy percent:", (nltk.classify.accuracy(voted_classifier ,D_test))*100)


# In[10]:


#15 most informative features of our dictionnary:
NB_classifier.show_most_informative_features(15)
#engrossing appears 21 times more in a neg review than a pos


# In[11]:


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

