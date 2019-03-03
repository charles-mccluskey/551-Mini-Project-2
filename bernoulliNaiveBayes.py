import re
from math import log
from collections import Counter


def get_features(text):
    """Extracts features from text

    Args:
        text (str): A blob of unstructured text (pre-processed review)
    """
    return set([w.lower() for w in text.split(" ")])


class BernoulliNBClassifier(object):

    def __init__(self):
        self._log_priors = None  # Class priors
        self._cond = None  # Conditional probabilities table
        self.fre_words = []  # List of most frequent words based on training set reviews
        self.num_words = 10000  # Number of selected most frequent words

    def train(self, reviews, labels):
        """Train a Bernoulli naive Bayes classifier

        Args:
            reviews (list): Each element in this list
                is a blog of text( a review)
            labels (list): The ground truth label for
                each review
        """

        """Compute log(P(Y))
        """
        label_counts = Counter(labels)
        N = float(sum(label_counts.values()))
        self._log_priors = {k: log(a / N) for k, a in label_counts.items()}

        """Find the most common words in the corpus
        """
        word_count = {}

        for i in range(len(reviews)):  # Run through training reviews
            for word in reviews[i].split():
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        word_counter = Counter(word_count)

        for word, count in word_counter.most_common(self.num_words):  # Append most frequent words to use as feature
            self.fre_words.append(word)

        # Split each review into words
        X = [set(get_features(r)) for r in reviews]

        """Compute log( P(X|Y) )
        """
        self._cond = {l: {f: 0. for f in self.fre_words} for l in self._log_priors}

        # Step through each review
        for x, l in zip(X, labels):
            for f in x:
                if f in self.fre_words:  # Checks whether the word in the training review is in the frequent words' list
                    self._cond[l][f] += 1.

        # Compute log probs
        for l in self._cond:
            N = label_counts[l]
            self._cond[l] = {f: (a + 1.) / (N + 2.) for f, a in self._cond[l].items()}

    def predict(self, text):
        """Make a prediction from text
        """
        x = get_features(text)
        pred_class = None
        max_m = float("-inf")

        # Perform MAP estimation
        for l in self._log_priors:
            log_sum = self._log_priors[l]
            for f in self.fre_words:
                prob = self._cond[l][f]
                log_sum += log(prob if f in x else 1. - prob)
            if log_sum > max_m:
                max_m = log_sum
                pred_class = l

        return pred_class


def make_bernoulli(reviews_train, labels_train, model_params):
    # Train model
    print('Number of training examples: {0}'.format(len(labels_train)))
    print('Training begins ...')
    my_nb = BernoulliNBClassifier()
    my_nb.train(reviews_train, labels_train)
    print('Training finished!')
    print('Number of features found: {0}'.format(len(my_nb.fre_words)))
    return my_nb

def error_bernoulli(model, X, Y):
    print('Testing model...')
    f = lambda doc, l: 1. if model.predict(doc) != l else 0.
    num_miss = sum([f(doc, l) for doc, l in zip(X, Y)])

    N = len(Y) * 1.
    error_rate = round(100. * (num_miss / N), 3)

    print('Error rate of {0}% ({1}/{2})'.format(error_rate, int(num_miss), int(N)))
    return error_rate

def preprocess(comments):
    rnp = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    rs = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    comments = [rnp.sub("", line.lower()) for line in comments]
    comments = [rs.sub(" ", line) for line in comments]
    return comments

def confusion_matrix_bernoulli(model, X, Y):
    pred = [model.predict(comment) for comment in X]
    print("Prediction done")
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