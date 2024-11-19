
import math
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations
from program.labelling_functions.ContainsWordLabeller import ContainsWordLabeller
from program.labelling_functions.RandomForestLabeller import RandomForestLabeller
from program.labelling_functions.EntityLabeller import EntityLabeller
from program.labelling_functions.WordEntityMixLabeller import WordEntityMixLabeller
from program.labelling_functions.DecisionTreeLabeller import DecisionTreeLabeller
from program.labelling_functions.KNeighborsLabeller import KNeighborsLabeller
from program.labelling_functions.LogisticRegressionLabeller import LogisticRegressionLabeller
from program.labelling_functions.SVMLabeller import SVMLabeller
from tqdm import tqdm

import program.run_config as rc


class LFGenerator:
    """
    Generate the Labelling Functions that will be then passed on to Snorkel.
    Given a set of labelled data, unlabelled data and the word counts, generate a
    set of labelling functions.
    """

    def __init__(self, training_x: list, training_y: list, testing: list, testing_y: list, intents: list, entity_model_path: str):
        """
        Initialize the generator with the necessary data.

        :param training_x: The training data.
        :param training_y: The training labels.
        :param testing: The testing data.
        """
        self.labelling_functions = []
        self.training_x = training_x
        self.training_y = training_y
        self.testing = testing
        self.testing_y = testing_y
        self.intents = intents
        self.entity_model_path = entity_model_path

        self.vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode')
        self.unigrams = self.vectorizer.fit_transform(self.training_x) # please why is self.testing used here?, could this be causing data leakage? (remove + self.testing)
        self.lfs = []

    def generate_lfs(self, max_words: int = 3) -> list:
        """
        Generate the labelling functions.
        Currently, only generating using the training data and seeing
        if the correctly labelled has words that the rest of the training data
        that is labelled as 0 does not have.

        :param max_words: The maximum amount of words in a single labelling function.
        :return: A list of labelling functions tuples (word, intent).
        """
        self.words(max_words)
        self.ml()
        self.entity()
        self.word_entity_mix()
        return self.lfs

    def words(self, cardinality: int = 3, threshold: float = rc.WORDS_LABELLER_THRESHOLD) -> None:
        """
        Generate labelling functions containing only words, either single or combinations.

        :param cardinality: Up to how big a combination of words to generate.
        """
        single_words = self.vectorizer.get_feature_names()

        for i in range(1, cardinality + 1):
            print("Generating word LF with cardinality: " + str(i))
            for words in tqdm(combinations(single_words, i)):
                lf = ContainsWordLabeller(words)
                lf.find_label(self.training_x, self.training_y, self.intents, threshold)
                if lf.label == -1:
                    continue
                self.lfs.append(lf)

    def ml(self) -> None:
        """
        Generate a random forest, 1 for each intent.

        :param n: Number of trees to generate.
        """
        from sklearn.metrics import accuracy_score

        for model in (RandomForestLabeller, DecisionTreeLabeller, KNeighborsLabeller, LogisticRegressionLabeller, SVMLabeller):
            lf = model(self.vectorizer)
            lf.dataset = (self.training_x, self.training_y)
            lf.number_of_queries = len(self.training_x)
            lf.pattern_basis = lf.dataset
            lf.fit(self.unigrams[:len(self.training_x)], self.training_y)

            predicted_labels = [lf.apply(sentence) for sentence in self.testing]
            accuracy = accuracy_score(self.testing_y, predicted_labels)

            print('ACCURACY', accuracy)
            
            self.lfs.append(lf)

    def entity(self, threshold: float = rc.ENTITY_LABELLER_THRESHOLD) -> None:
        """
        Generate labelling functions based on distinct entity types, for each intent.

        :param threshold: This value determines how unique an entity must be to a label to be considered 'unique'.
        """
        entity_label_map = EntityLabeller.find_entity_unique_class(self.training_x, self.training_y, self.entity_model_path, threshold)
        # print("ENTITY LABEL MAP")
        # print(entity_label_map)

        for key in entity_label_map.keys():
            lf = EntityLabeller(key)
            lf.find_entity_label(key)
            if lf.label == -1:
                continue
            self.lfs.append(lf)

    def word_entity_mix(self, threshold: float = rc.WORDS_ENTITY_LABELLER_THRESHOLD) -> None:
        """
        Generate labelling functions based on mix of distinct entity types, for each intent and distinct words.

        """
        entity_label_map = EntityLabeller.find_entity_unique_class(self.training_x, self.training_y, self.entity_model_path, threshold)
        single_words = self.vectorizer.get_feature_names()

        word_entity_label_map = WordEntityMixLabeller.find_word_entity_unique_class(self.training_x, self.training_y, single_words, entity_label_map.keys(), threshold)

        for key in word_entity_label_map.keys():
            lf = WordEntityMixLabeller(key)
            lf.find_word_entity_label(key)
            if lf.label == -1:
                continue
            self.lfs.append(lf)

    @staticmethod
    def generate_domain_file(training_data_path, domain_file_path):
        # Load the training data
        with open(training_data_path, 'r') as f:
            training_data = yaml.safe_load(f)

        # Extract the intents and entities
        intents = [item['intent'] for item in training_data['nlu'] if 'intent' in item]
        entities = set()
        for item in training_data['nlu']:
            if 'intent' in item:  # Check if 'intent' exists in item
                for example in item['examples'].split('\n'):
                    for word in example.split():
                        if word.startswith('[') and ']' in word:
                            entities.add(word[word.index('[')+1 : word.index(']')])

        # Create the domain data
        domain_data = {
            'intents': intents,
            'entities': list(entities),
            'responses': {},
            'actions': []
        }

        # Write the domain data to the domain file
        with open(domain_file_path, 'w') as f:
            yaml.dump(domain_data, f)
