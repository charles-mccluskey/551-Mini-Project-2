from Project2_text_processing import getAllData, openComment
from sklearn.linear_model import LogisticRegression
import numpy as np
from validation import k_fold_cross_validation, held_out_validation_set
import bernoulliNaiveBayes
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import StandardScaler


def model_pipeline(X, Y, model_params):
    if (model_params["name"] == 'logistic'):
        if model_params["bin"]:
            pclf = Pipeline([
                ('vect', CountVectorizer(stop_words='english', max_features=model_params["num_words"],
                                     ngram_range=model_params['ngram'], binary=model_params['bin'])),
                ('norm', Normalizer()),
                ('clf', LogisticRegression(solver='lbfgs')),
            ])
        else:
            pclf = Pipeline([
                ('vect', CountVectorizer(stop_words='english', max_features=model_params["num_words"],
                                         ngram_range=model_params['ngram'], binary=model_params['bin'])),
                ('tfidf', TfidfTransformer()),
                ('norm', Normalizer()),
                ('clf', LogisticRegression(solver='lbfgs')),
            ])
    elif (model_params["name"] == 'tree'):
        pclf = Pipeline([
            ('vect', CountVectorizer(stop_words='english', max_features=model_params["num_words"],
                                     ngram_range=model_params['ngram'], binary=model_params['bin'])),
            ('tfidf', TfidfTransformer()),
            ('norm', Normalizer()),
            ('clf', tree.DecisionTreeClassifier(max_depth=model_params['depth'])),
        ])
    elif((model_params["name"] == 'svm')) :
        pclf = Pipeline([
            ('vect', CountVectorizer(stop_words='english', max_features=model_params["num_words"], ngram_range=model_params['ngram'] , binary = model_params['bin'] )),
            ('tfidf', TfidfTransformer()),
            ('norm', Normalizer()),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', svm.SVC(max_iter=model_params["iter"], gamma="scale", probability=model_params["prob"])),
        ])
    elif ((model_params["name"] == 'gnb')):
        pclf = Pipeline([
            ('vect', CountVectorizer(stop_words='english', max_features=model_params["num_words"], ngram_range=model_params['ngram'] , binary = model_params['bin'] )),
            # ('tfidf', TfidfTransformer()),
            ('norm', Normalizer()),
            ('clf', MultinomialNB()),
            ])
    elif ((model_params["name"] == 'stacklog')):
        pclf = StackingLogistic(model_params)
    pclf.fit(X, Y)
    # print("xddqsdfd")
    return pclf

def class_to_num(X):
    return [1 if x == "pos" else 0 for x in X]

def check_scores(sid, X):
    toRet = []
    for comment in X:
        ss = sid.polarity_scores(comment)
        toRet.append(1 if ss["pos"]>=ss["neg"] else 0)
    return toRet

def nltk_scores(sid, X):
    toRet = []
    for i in range(len(X)):
        ss = sid.polarity_scores(X[i])
        toRet.append([ss["neg"], ss["pos"]])
    return toRet


class StackingLogistic:
    def __init__(self, model_params):
        self.sid = SentimentIntensityAnalyzer()
        self.params = model_params

    def transform(self, X):
        pred_log = self.logistic.predict_proba(X)
        pred_tree = self.tree.predict_proba(X)
        pred_svm = self.svm.predict_proba(X)
        pred_gnb = self.gnb.predict_proba(X)
        pred_nltk = nltk_scores(self.sid, X)
        return np.concatenate((pred_log, pred_tree, pred_svm, pred_gnb, pred_nltk), axis = 1)

    def fit_classifiers(self, X, Y):
        self.params["name"] = 'logistic'
        self.logistic = model_pipeline(X, Y, self.params)
        self.params["name"] = 'tree'
        self.params["num_words"] = 75000
        self.params["bin"]=False
        self.tree = model_pipeline(X, Y, self.params)
        self.params["num_words"] = 20000
        self.params["name"] = 'svm'
        self.svm = model_pipeline(X, Y, self.params)
        self.params["bin"] = True
        self.params["name"] = 'gnb'
        self.gnb = model_pipeline(X, Y, self.params)
        self.params["name"] = "stacklog"

    def fit(self, X, Y):
        mid = int(len(Y)/2)
        self.fit_classifiers( X[:mid], Y[:mid])
        a = self.transform(X[mid:])
        self.fit_classifiers(X[mid:], Y[mid:])
        b = self.transform(X[:mid])
        self.predictor  = LogisticRegression(solver='liblinear')
        self.predictor.fit(np.concatenate((a,b)), Y)
        self.params["prob"] = True
        self.fit_classifiers(X, Y)

    def predict(self, X):
        pred = self.transform(X)
        return self.predictor.predict(pred)




def error(model , X, Y):
    pred = model.predict(X)
    err = 0
    # print('Getting error for',len(X), 'samples')
    for i in range(len(Y)):
        if pred[i]!=Y[i]:
            err +=1
    # print(err, 'incorrectly classified samples')
    return (len(Y)-err)/len(Y)



def confusion_matrix(model, X, Y):
    pred = model.predict(X)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(Y)):
        if Y[i] == 'pos':
            if pred[i] == 'pos':
                true_pos +=1
            else :
                false_neg +=1
        else:
            if pred[i] == 'neg':
                true_neg += 1
            else:
                false_pos +=1
    return [[true_pos/len(Y), false_neg/len(Y), false_pos/len(Y), true_neg/len(Y)]]

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

def shuffle_files(Apos, Aneg, Bpos, Bneg):
    X = [val for pair in zip(Apos, Aneg) for val in pair]
    Y = [val for pair in zip(Bpos, Bneg) for val in pair]
    return (X,Y)

def present_results(name, results):
    training = results[0]
    validation = results[1]
    training_accuracy = training[0]+training[3]
    validation_accuracy = validation[0]+validation[3]
    print("For", name)
    print("Accuracy on training set:", training_accuracy)
    print("Accuracy on validation set:", validation_accuracy)
    print("Confusion matrix:")
    print("True positives", validation[0], "   ", "False negatives", validation[1])
    print("False positives", validation[2], "   ", "True negatives", validation[3])


if __name__ == '__main__':
    res = getAllData()
    Apos = [openComment(res, "pos", i) for i in range(12500)]
    Aneg = [openComment(res, "neg", i) for i in range(12500)]
    Bpos = ["pos" for i in range(12500)]
    Bneg = ["neg" for i in range(12500)]
    (X, Y) = shuffle_files(Apos, Aneg, Bpos, Bneg)
    model_params = {"name": 'logistic', "bin": False, 'num_words': 20000, 'ngram': (1, 3)}

    # Logistic regression with tfidf scores : should take 5 minutes
    present_results("LogisticRegression with tfidf",
                    k_fold_cross_validation(X, Y, 5, model_pipeline, model_params, confusion_matrix, confusion=True))
    model_params["bin"] = True

    # Logistic regression with binary word counts : should take 5 minutes
    present_results("logistic regression with binary word counts",
                    k_fold_cross_validation(X, Y, 5, model_pipeline, model_params, confusion_matrix, confusion=True))
    model_params["name"] = "tree"
    model_params["depth"] = 20
    model_params["bin"] = False
    model_params["num_words"] = 75000

    # Decision tree classifier  : should take 5 minutes
    present_results("sklearn decision tree",
                    k_fold_cross_validation(X, Y, 5, model_pipeline, model_params, confusion_matrix, confusion=True))

    model_params["name"] = "stacklog"
    model_params["iter"] = 1000
    model_params["bin"] = True
    model_params["depth"] = 20
    model_params["num_words"] = 20000
    model_params['prob'] = True

    # Stacking model : Takes around an hour with 5-fold cross validation : change the 5 to a lower to test faster
    present_results("stacking with logistic regression",
                    k_fold_cross_validation(X, Y, 5, model_pipeline, model_params, confusion_matrix, confusion=True))

    bernoulli_comments = bernoulliNaiveBayes.preprocess(X)

    # Our implementation of Bernoulli : Takes around 40 minutes with 5-fold cross validation : change the 5 to a lower to test faster
    present_results("our version of Bernoulli Naive Bayes",
                    k_fold_cross_validation(bernoulli_comments, Y, 5, bernoulliNaiveBayes.make_bernoulli, model_params,
                                            bernoulliNaiveBayes.confusion_matrix_bernoulli, confusion=True))

