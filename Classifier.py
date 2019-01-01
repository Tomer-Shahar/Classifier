"""
The class for part 2 of the assignment
Builds a classification model and classifies a given file.
"""
import warnings
from time import time

from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline




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

        self.__clf_list = ((SGDClassifier(), "SVM"), (Perceptron(), "Perceptron"))
        self.__clf_funcs = [SGDClassifier, Perceptron]
        self.__model_list = (("tf_idf", self.tf_idf_feature_extraction), ("bigram", self.bigram_feature_extraction))
        self.__vect_funcs = (('tf_idf', TfidfVectorizer), ('bigram', CountVectorizer))

    def tf_idf_feature_extraction(self):
        """
        Extract features with tf-idf
        :param data: training data to extract features
        :return: vector of the features
        """
        print('=' * 80)
        print("TF-IDF Feature Extraction")
        t0 = time()
        vectorizer = TfidfVectorizer()
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
        bigram_vectorizer = CountVectorizer()
        bigram_train = bigram_vectorizer.fit_transform(self.training_data.data)
        bigram_test = bigram_vectorizer.transform(self.test_data.data)
        duration = time() - t0
        print("DONE!!! total time: %fs" % duration)
        print('=' * 80)
        return bigram_train, bigram_test

    def train_model(self):
        """
        :return:
        """
        for model_name, model_function in self.__model_list:
            print(model_name)
            scores = []
            train, test = model_function()
            for clf, name in self.__clf_list:
                print('=' * 80)
                print(name)
                scores.append(self.classification(clf, train, test))
            self.best_score(scores)

    def train_optimise(self):
        training_scores = []
        best_params_list = []
        for vect_name, vect_func in self.__vect_funcs:
            print(vect_name)
            for clf_func, clf_name in self.__clf_list:
                print('=' * 80)
                print(clf_name)
                t_score, best_params = self.optimize(vect_func, clf_func)
                training_scores.extend([t_score])
                best_params_list.extend([best_params])
        vect_index, clf_index, best_index = self.best_score(training_scores)
        self.run_best_model(vect_index, clf_index, best_params_list[best_index])

    def run_best_model(self, vect_index, clf_index, best_params):
        vect_name, vect_func = self.__vect_funcs[vect_index]
        vect = vect_func()
        print('=' * 80)
        print(str(vect_name) + " Feature Extraction")
        t0 = time()
        vec_train = vect.fit_transform(self.training_data.data)
        vec_test = vect.transform(self.test_data.data)
        duration = time() - t0
        print("DONE!!! total time: %fs" % duration)
        print('=' * 80)
        clf_func = self.__clf_funcs[clf_index]
        clf = clf_func(alpha=best_params.get('clf__alpha'), penalty=best_params.get('clf__penalty'))
        print('Training:')
        print(clf)
        t0_training = time()
        clf.fit(vec_train, self.training_data.target)
        train_time = time() - t0_training
        print("train time: %0.3fs" % train_time)
        t0_test = time()
        pred = clf.predict(vec_test)
        test_time = time() - t0_test
        print("test time: %0.3fs" % test_time)
        score = metrics.accuracy_score(self.test_data.target, pred)
        print('accuracy:  ' + str(score))

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

        t0 = time()
        pred = clf.predict(model_test_data)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.test_data.target, pred)
        print("accuracy:   " + str(score))
        return score

    @staticmethod
    def best_score(scores):
        best_index = scores.index(max(scores))
        print("The best classification for this corpus is:")
        if 0 == best_index:
            print("feature extraction: tf-idf, classification: SVM")
            print(str(scores[best_index]))
            return 0, 0, best_index
        if 1 == best_index:
            print("feature extraction: tf-idf, classification: Perceptron")
            print(str(scores[best_index]))
            return 0, 1, best_index
        if 2 == best_index:
            print("feature extraction: tf-idf, classification: Naive Base")
            print(str(scores[best_index]))
            return 0, 2, best_index
        if 3 == best_index:
            print("feature extraction: bigram, classification: SVM")
            print(str(scores[best_index]))
            return 1, 0, best_index
        if 4 == best_index:
            print("feature extraction: bigram, classification: Perceptron")
            print(str(scores[best_index]))
            return 1, 1, best_index
        if 5 == best_index:
            print("feature extraction: bigram, classification: Naive Base")
            print(str(scores[best_index]))
            return 1, 2, best_index

    def optimize(self, feature_func, classifier_func):
        nb_clf = Pipeline([('vect', feature_func()),
                           ('clf', classifier_func)])
        parameters = {
            'clf__alpha': [0.001, 0.0001],
            'clf__penalty': [None, 'l2', 'l1', 'elasticnet'],
        }
        gs_clf = GridSearchCV(nb_clf, parameters)
        gs_clf = gs_clf.fit(self.training_data.data, self.training_data.target)
        print("Best parameters: " + str(gs_clf.best_params_))
        print('Best score: ' + str(gs_clf.best_score_))
        training_score = gs_clf.best_score_
        return training_score, gs_clf.best_params_
