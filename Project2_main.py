from sklearn.feature_extraction.text import TfidfVectorizer
from Project2_text_processing import getAllData, openComment
from sklearn.linear_model import LogisticRegression
import numpy as np
import numbers
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

def tfidf(X):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features = 10)
    x = vectorizer.fit_transform(X)
    return x.toarray()




def make_model2(reviews_train, labels_train):

    # Train model
    print('Number of training examples: {0}'.format(len(labels_train)))
    print('Training begins ...')
    my_nb = BernoulliNBClassifier()
    my_nb.train(reviews_train, labels_train)
    print('Training finished!')
    print('Number of features found: {0}'.format(len(my_nb.fre_words)))
    return my_nb

def error_fun2(model, X, Y):
    print('Testing model...')
    f = lambda doc, l: 1. if model.predict(doc) != l else 0.
    num_miss = sum([f(doc, l) for doc, l in zip(X, Y)])

    N = len(Y) * 1.
    error_rate = round(100. * (num_miss / N), 3)

    print('Error rate of {0}% ({1}/{2})'.format(error_rate, int(num_miss), int(N)))
    return error_rate


def model_pipeline(X, Y):
    pclf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        ('clf', MultinomialNB()),
    ])
    pclf.fit(X, Y)
    return pclf

def error_fun(model, X, Y):
    y_pred = model.predict(X)
    err = 0
    for i in range(len(Y)):
        if(Y[i] != y_pred[i]):
           err += 1
    return err/len(Y)

if __name__ == '__main__':
    res = getAllData()
    X = [openComment(res,"pos",i) for i in range(12500)]+[openComment(res,"neg",i) for i in range(12500)]
    Y = ["pos" for i in range(12500)]+["neg" for i in range(12500)]
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



    rnp = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    rs = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    X = [rnp.sub("", line.lower()) for line in X]
    X = [rs.sub(" ", line) for line in X]
    print(k_fold_cross_validation(X, Y, 5, make_model2, error_fun2))

