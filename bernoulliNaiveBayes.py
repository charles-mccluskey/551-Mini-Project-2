import re
from math import log
import glob
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