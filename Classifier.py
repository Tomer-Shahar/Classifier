"""
The class for part 2 of the assignment
Builds a classification model and classifies a given file.
"""
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB


class Classifier:

    def __init__(self, corpus_path=None, train_data=None, target_data=None):
        
        if train_data:
            self.training_data = train_data
        else:
            self.training_data = load_files(corpus_path + '\\training', encoding='utf-8')

        if target_data:
            self.test_data = target_data
        else:
            self.test_data = load_files(corpus_path + '\\test', encoding='utf-8')

        self.__clf_list = ((SGDClassifier(), "SVM"), (Perceptron(), "Perceptron"), (MultinomialNB(), "Naive Bayes"))
        self.__model_list = (("tf_idf", self.tf_idf_feature_extraction), ("bigram", self.bigram_feature_extraction), ("hash", self.hash_feature_extraction))


    def tf_idf_feature_extraction(self):
        """
        Extract features with tf-idf
        :param data: training data to extract features
        :return: vector of the features
        """
        print('=' * 80)
        print("TF-IDF Feature Extraction")
        t0 = time()
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        vec_train = vectorizer.fit_transform(self.training_data.data)
        vec_test = vectorizer.transform(self.test_data.data)
        duration = time() - t0
        print("DONE!!!!! total time: %fs" % duration)
        print('=' * 80)
        return vec_train, vec_test

    def bigram_feature_extraction(self):
        """
        Extract features with bigram
        :return: vector with the features
        """
        print('=' * 80)
        print('Bigram Feature Extraction')
        t0 = time()
        bigram_vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words='english')
        bigram_train = bigram_vectorizer.fit_transform(self.training_data.data)
        bigram_test = bigram_vectorizer.transform(self.test_data.data)
        duration = time() - t0
        print("DONE!!! total time: %fs" % duration)
        print('=' * 80)
        return bigram_train, bigram_test

    def hash_feature_extraction(self):
        print('=' * 80)
        print("Hash Feature Extraction")
        t0 = time()
        hash_vectorizer = HashingVectorizer(non_negative=True)
        hash_train = hash_vectorizer.fit_transform(self.training_data.data)
        hash_test = hash_vectorizer.transform(self.test_data.data)
        duration = time() - t0
        print("DONE!!! total time: %fs" % duration)
        print('=' * 80)
        return hash_train, hash_test

    def train_model(self):
        """
        :return:
        """
        results = []
        scores = []
        for model_name, model_function in self.__model_list:
            print(model_name)
            train, test = model_function()
            for clf, name in self.__clf_list:
                print('=' * 80)
                print(name)
                results.append(self.classification(clf, train, test))
            scores = scores + self.plot_compare(results, name=model_name)
            results = []
        self.bestScore(scores)

    def classification(self, clf, model_train_data, model_test_data):
        """
        Training and classification with given model
        :param model_test_data:
        :param model_train_data:
        :param clf: method to classify with
        :return:
        """
        print('=' * 80)
        print('Training:')
        print(clf)
        t0_training = time()
        clf.fit(model_train_data, self.training_data.target)
        train_time = time() - t0_training
        print("train time: %0.3fs" % train_time)

        t0_test = time()
        prediction = clf.predict(model_test_data)
        test_time = time() - t0_test
        print("test time: %0.3fs" % test_time)

        score = metrics.accuracy_score(self.test_data.target, prediction)
        print("the score is: %0.3f" % score)
        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time

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

    @staticmethod
    def plot_compare(results, name=""):
        scores = []
        print(name)
        indices = np.arange(len(results))
        results = [[x[i] for x in results] for i in range(4)]
        clf_names, score, training_time, test_time = results
        for s in score:
            scores.append(s)
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)
        plt.figure(figsize=(12, 8))
        plt.title(name + "Score")
        plt.barh(indices, score, .2, label="score", color="navy")
        plt.barh(indices + .3, training_time, .2, label="training time", color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color="darkorange")
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)
        plt.show()
        return scores

    def bestScore(self, scores):
        best_index = scores.index(max(scores))
        print("The best classification for this corpus is:")
        if 0 == best_index:
            print("feature extraction: tf-idf, classification: SVM")
        if 1 == best_index:
            print("feature extraction: tf-idf, classification: Perceptron")
        if 2 == best_index:
            print("feature extraction: tf-idf, classification: Naive Base")
        if 3 == best_index:
            print("feature extraction: bigram, classification: SVM")
        if 4 == best_index:
            print("feature extraction: bigram, classification: Perceptron")
        if 5 == best_index:
            print("feature extraction: bigram, classification: Naive Base")
        if 6 == best_index:
            print("feature extraction: hash, classification: SVM")
        if 7 == best_index:
            print("feature extraction: hash, classification: Perceptron")
        if 8 == best_index:
            print("feature extraction: hash, classification: Naive Base")
