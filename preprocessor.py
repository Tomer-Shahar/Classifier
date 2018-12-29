from Parser import Parser
import os
import json
import matplotlib.pyplot as plt
import math


class PreProcessor:
    """
    This class performs part 1 of the assignment.
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.category_num = -1
        self.num_of_docs_per_category = {}
        self.category_to_terms_map = {}  # terms in each category
        self.top_ten_terms_per_category = {}
        self.term_appearances_per_category = {}  # In how many docs has each term appeared (per category)
        self.parser = Parser()
        self.categories = {
            'C01': 'Bacterial Infections and Mycoses',
            'C02': 'Virus Diseases',
            'C03': 'Parasitic Diseases',
            'C04': 'Neoplasms',
            'C05': 'Musculoskeletal Diseases',
            'C06': 'Digestive System Diseases',
            'C07': 'Stomatognathic Diseases',
            'C08': 'Respiratory Tract Diseases',
            'C09': 'Otorhinolaryngologic Diseases',
            'C10': 'Nervous System Diseases',
            'C11': 'Eye Diseases',
            'C12': 'Urologic and Male Genital Diseases',
            'C13': 'Female Genital Diseases and Pregnancy Complications',
            'C14': 'Cardiovascular Diseases',
            'C15': 'Hemic and Lymphatic Diseases',
            'C16': 'Neonatal Diseases and Abnormalities',
            'C17': 'Skin and Connective Tissue Diseases',
            'C18': 'Nutritional and Metabolic Diseases',
            'C19': 'Endocrine Diseases',
            'C20': 'Immunologic Diseases',
            'C21': 'Disorders of Environmental Origin',
            'C22': 'Animal Diseases',
            'C23': 'Pathological Conditions, Signs and Symptoms'
        }

        for category, name in self.categories.items():
            self.top_ten_terms_per_category[category] = []
            self.category_to_terms_map[category] = {}
            self.term_appearances_per_category[category] = {}

    def process_all_files(self):

        if not os.path.exists(os.path.join(self.output_path, '\\training')):
            os.makedirs(os.path.join(self.output_path, '\\training'))

        self.output_parsed_files(self.input_path + '\\training', self.output_path + '\\training', collect_data=True)
        self.output_parsed_files(self.input_path + '\\test', self.output_path + '\\test', collect_data=False)

    def output_parsed_files(self, input_path, output_path, collect_data):

        for root, dirs, corpus_files in os.walk(input_path):
            self.category_num = len(dirs)
            for folder in dirs:  # Walks each category folder.
                for category_folder, directories, files in os.walk(os.path.join(input_path, folder)):
                    for file in files:
                        text_file = open(os.path.join(category_folder, file), 'r').read()
                        parsed_text_file, word_dict = self.parser.parse_text_file(text_file, collect_data)

                        if collect_data:
                            self.num_of_docs_per_category[folder] = len(files)
                            self.merge_word_dicts(folder, word_dict)
                            self.update_term_appearances(folder, word_dict)

                        self.write_file(parsed_text_file, output_path, folder, file + '_parsed')
            break

        if collect_data:
            self.print_statistics()

    def print_statistics(self):
        """
        Print data regarding the parsed files.
        """
        self.write_statistics()
        print("Number of categories: " + str(self.category_num))
        print("Number of documents per category:")
        i = 1
        for cat_id, num in self.num_of_docs_per_category.items():
            print(str(i) + '. ' + self.categories[cat_id] + ": " + str(num))
            i += 1

        print()
        print("Top 10 Words Per Category: ")
        print()

        for category, top_ten in self.top_ten_terms_per_category.items():
            print(self.categories[category] + ":")
            for i in range(len(top_ten)):
                print(str(i+1) + ". \'" + top_ten[i][0] + "\' appeared " + str(top_ten[i][1]) + " times")

            print()

        self.generate_histograms()

    def generate_histograms(self):
        for category, value in self.term_appearances_per_category.items():
            category_words = [(k, value[k]) for k in sorted(value, key=value.get, reverse=True)]
            num_bins = category_words[0][1]
            appearances = []
            for term, num in value.items():
                appearances.append(num)
            # plt.figure(figsize=(10, 10))
            plt.hist(appearances, num_bins, facecolor='xkcd:dusky blue', rwidth=1)
            plt.title("Histogram of terms appearing in the category: \n" + self.categories[category])
            plt.xlabel('Number of documents that a term appeared in')
            plt.ylabel('Number of terms that appeared in X documents')
            plt.grid()
            plt.show()

    def write_statistics(self):

        statistics_path = os.path.join(self.output_path, 'statistics')
        if not os.path.exists(statistics_path):
            os.makedirs(statistics_path)
        with open(os.path.join(statistics_path, 'corpus_data.json'), 'w+') as new_file:
            info_dict = {
                'category_num': self.category_num,
                'num_of_docs_per_category': self.num_of_docs_per_category
            }
            json.dump(info_dict, new_file)
        terms_distribution_dict = {}

        for category, value in self.category_to_terms_map.items():
            category_words = [(k, value[k]) for k in sorted(value, key=value.get, reverse=True)]
            terms_distribution_dict[category] = category_words
            self.top_ten_terms_per_category[category] = category_words[:10]

        dist_dict = {
            'term_appearances_per_category': self.term_appearances_per_category,
            'top_ten_terms_per_category': self.top_ten_terms_per_category,
            'category_to_terms_map': self.category_to_terms_map
        }
        with open(os.path.join(statistics_path, 'distribution_data.json'), 'w+') as new_file:
            json.dump(dist_dict, new_file)

    def load_statistics(self, statistics_path):
        with open(os.path.join(statistics_path, 'corpus_data.json'), 'r') as corpus_json:
            corpus_info = json.load(corpus_json)
            self.category_num = corpus_info['category_num']
            self.num_of_docs_per_category = corpus_info['num_of_docs_per_category']

        with open(os.path.join(statistics_path, 'distribution_data.json'), 'r') as corpus_json:
            dist_dict = json.load(corpus_json)
            self.category_to_terms_map = dist_dict['category_to_terms_map']
            self.top_ten_terms_per_category = dist_dict['top_ten_terms_per_category']
            self.term_appearances_per_category = dist_dict['term_appearances_per_category']

    @staticmethod
    def write_file(text_list, output_path, folder, file_name):

        path = os.path.join(output_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, file_name), 'w+') as new_file:
            new_file.write(' '.join(text_list))

    def merge_word_dicts(self, category_folder, word_dict):

        for word, val in word_dict.items():
            if word in self.category_to_terms_map[category_folder]:
                self.category_to_terms_map[category_folder][word] += val
            else:
                self.category_to_terms_map[category_folder][word] = word_dict[word]

    def update_term_appearances(self, category, word_dict):

        for word, num in word_dict.items():
            if word in self.term_appearances_per_category[category]:
                self.term_appearances_per_category[category][word] += 1
            else:
                self.term_appearances_per_category[category][word] = 1
