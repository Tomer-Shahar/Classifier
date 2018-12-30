"""
The main class that will run the different modules of the assignment
"""
from preprocessor import PreProcessor
from Classifier import Classifier

if __name__ == '__main__':
    # corpus_path = '.\\ohsumed-first-20000-docs'  # PATH TO THE CORPUS FOLDER THAT CONTAINS THE SUB-FOLDERS
    # output_path = '.\\output'  # Output path where the statistics and parsed files will be saved

    # prep = PreProcessor(corpus_path, output_path)
    # prep.process_all_files()

    # prep.load_statistics('.\\statistics')
    # prep.print_statistics()

    classifer = Classifier(courpus_path='D:\\ohsumed-first-20000-docs')
    tf_idf_train, tf_idf_test = classifer.tf_idf_feature_extraction()
    bigram_train, bigram_test = classifer.bigram_feature_extraction()
    hash_train, hash_test = classifer.hash_feature_extraction()

