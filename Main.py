"""
The main class that will run the different modules of the assignment
"""
import warnings

from Classifier import Classifier
from preprocessor import PreProcessor

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    corpus_path = '.\\ohsumed-first-20000-docs'  # PATH TO THE CORPUS FOLDER THAT CONTAINS THE SUB-FOLDERS
    output_path = '.\\output'  # Output path where the statistics and parsed files will be saved

    #prep = PreProcessor(corpus_path, output_path)
    #prep.process_all_files()
    #prep.load_statistics('.\\output\\statistics')
    #prep.print_statistics()

    classifier = Classifier(corpus_path=corpus_path)
    classifier.train_model()
    classifier.train_optimise()
