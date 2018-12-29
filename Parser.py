import nltk

from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import re

#nltk.download("stopwords")

stemmer = EnglishStemmer()


class Parser:

    def __init__(self):

        stop_words = set(stopwords.words("english"))
        stop_word_file = self.create_stop_words_set()
        self.stop_words = stop_word_file | stop_words

    def parse_text_file(self, text, collect_data):
        """
        :param collect_data: a bool to decide if we want data about the text file or not.
        :param text: text file to parse, a string
        :return: returns a clean, parsed string and a dictionary containing the terms found and their frequency
        """
        #split_text = word_tokenize(text)
        split_text = text.split(' ')
        parsed_text = []
        word_dict = {}

        for i in range(len(split_text)):
            word = self.normalize_word(split_text[i])
            if word in self.stop_words or len(word) == 0:
                continue
            word = self.parse_word(word)
            parsed_text.append(word)

            if collect_data:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        return parsed_text, word_dict

    @staticmethod
    def normalize_word(word):

        word = word.lower()
        word = word.replace('\n', '').replace('.', '')
        word = re.sub(r'[!@#|`$=)(\]\"\'\\[{/}^;:*+\-,?_><~]', '', word)
        return word

    @staticmethod
    def parse_word(word):
        if '%' == word[-1]:
            return word[:-1] + '_percent'

        parsed_word = stemmer.stem(word)

        return parsed_word

    @staticmethod
    def create_stop_words_set():
        path = './stop_words.txt'
        text_file = open(path, 'r', encoding="ISO-8859-1")
        stop_words_array = text_file.readlines()
        stop_words_set = set()

        for i in range(len(stop_words_array)):
            stop_words_set.add(stop_words_array[i].replace('\n', ''))

        return stop_words_set

