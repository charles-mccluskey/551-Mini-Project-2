from sklearn.feature_extraction.text import TfidfVectorizer
from Project2_text_processing import getAllData, openComment
from sklearn.linear_model import LogisticRegression
import numpy as np
import numbers
import collections
from validation import k_fold_cross_validation, held_out_validation_set
from bernoulliNaiveBayes import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from sklearn import svm
from sklearn import tree
from random import shuffle
from sklearn.pipeline import FeatureUnion


def model_pipeline(X, Y, model_params):
    if (model_params["name"] == 'logistic'):
        print('sweet')
        pclf = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', binary=model_params['bin'], stop_words='english', norm='l2',
                                      max_features=model_params["num_words"], ngram_range=model_params['ngram'])),
            ('clf', LogisticRegression(solver='lbfgs')),
        ])
    elif (model_params["name"] == 'tree'):
        pclf = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', binary = model_params['bin'] ,stop_words='english', norm='l2', max_features=model_params["num_words"], ngram_range=model_params['ngram'])),
            ('clf', tree.DecisionTreeClassifier()),
        ])
    else :
        pclf = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', binary = model_params['bin'] ,stop_words='english', norm='l2', max_features=model_params["num_words"], ngram_range=model_params['ngram'])),
            ('clf', svm.SVC(gamma='scale')),
        ])
    pclf.fit(X, Y)
    return pclf


def error(model , X, Y):
    pred = model.predict(X)
    err = 0
    print('Getting error for',len(X), 'samples')
    for i in range(len(Y)):
        if pred[i]!=Y[i]:
            err +=1
    print(err, 'incorretly classified samples')
    return (len(Y)-err)/len(Y)

def test_set(model):
    files = ['C:\\Users\\grego\\Documents\\test\\test\\'+str(i)+'.txt' for i in range(25000)]
    comments = []
    for i in range(25000):
        with open(files[i], 'rb') as f:
            comments.append(f.read().decode('utf-8', errors='ignore'))
    return model.predict(comments)

def write_results(filename, pred):
    with open(filename, 'w') as f:
        f.write('Id,Category\n')
        for i in range(25000):
            if pred[i] == 'pos':
                f.write(str(i) + ',1\n')
            else:
                f.write(str(i) + ',0\n')

def shuffle_files(A, B):
    indexes = list(range(25000))
    shuffle(indexes)
    X = []
    Y = []
    for i in range(25000):
        X.append(A[indexes[i]])
        Y.append(B[indexes[i]])
    return (X,Y)

if __name__ == '__main__':
    res = getAllData()
    A = [openComment(res,"pos",i) for i in range(12500)]+[openComment(res,"neg",i) for i in range(12500)]
    z = ["pos" for i in range(12500)]+["neg" for i in range(12500)]

    model_params = {"name":'logistic', "bin":False, 'num_words':20000, 'ngram' : (1,1) }
    (X, Y) = shuffle_files(A, z)
    print(k_fold_cross_validation(X, Y, 5, model_pipeline, model_params,  error))


    # print(collections.Counter(pred)['neg'])


    # mod = model_pipeline(v[5000:], z[5000:])
    # print(error(mod, v[:5000], z[20000:]))
    '''
    sentences = []
    for comment in X[0:10]:
        sentences.extend(tokenize.sent_tokenize(comment))
    sid = SentimentIntensityAnalyzer()
    for sentence in sentences:
        print(sentence)
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='')
            print()
    '''
    # n_instances = 100
    # subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    # obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
    # print(subj_docs[0])

    # res = getAllData()
    # X = [openComment(res,"pos",i) for i in range(12500)]+[openComment(res,"neg",i) for i in range(12500)]
    # Y = [1 for i in range(12500)]+[0 for i in range(12500)]


