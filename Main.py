"""
The main class that will run the different modules of the assignment
"""
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB

from preprocessor import PreProcessor
from Classifier import Classifier

if __name__ == '__main__':
    # corpus_path = '.\\ohsumed-first-20000-docs'  # PATH TO THE CORPUS FOLDER THAT CONTAINS THE SUB-FOLDERS
    # output_path = '.\\output'  # Output path where the statistics and parsed files will be saved

    # prep = PreProcessor(corpus_path, output_path)
    # prep.process_all_files()

    # prep.load_statistics('.\\statistics')
    # prep.print_statistics()

    clf_list = ((SGDClassifier(), "SVM"), (Perceptron(), "Perceptron"), (MultinomialNB(), "Naive Bayes"))

    classifer = Classifier(corpus_path='.\\ohsumed-first-20000-docs')
    classifer.train_model()


