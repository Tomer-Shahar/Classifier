"""
The class for part 2 of the assignment
Builds a classification model and classifies a given file.
"""
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB


class Classifier:

    def __init__(self, train_data_path=None, train_data=None, target_data=None):
        if train_data:
            self.training_data = train_data
        else:
            self.training_data = self.read_train_data(train_data_path)

        if target_data:
            self.target_data = target_data

    def tf_idf_feature_extraction(self, data):
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        vec_train = vectorizer.fit_transform(data)
        return vec_train

    def count_feature_extraction(self, data):
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        bigram_vectorizer.fit(data)
        return bigram_vectorizer

    def train_model(self, train_data):
        """
        :param train_data:
        :return:
        """

    @staticmethod
    def read_train_data(train_data_path):

        train_data = []
        for root, dirs, corpus_files in os.walk(train_data_path):
            for folder in dirs:  # Walks each category folder.
                for category_folder, directories, files in os.walk(os.path.join(train_data_path, folder)):
                    for file in files:
                        text_file = open(os.path.join(category_folder, file), 'r').read()
                        train_data.append(text_file)

            break

        return train_data
