import itertools
from collections import defaultdict
from typing import List, Dict, Union
from program.labelling_functions.LabellingFunction import LabellingFunction
from data.entity import EntityDetector

class WordEntityMixLabeller(LabellingFunction):
    entity_word_label_map = {}
    entity_detector = EntityDetector()
    word_list = []  # List of words that you want to consider
    entity_list = []  # List of entities that you want to consider
    threshold: float = 0.8
    number_of_queries_map = {}

    def __init__(self, key):
        super().__init__(f"wordentl{', '.join(key)}", self.apply)
        self.label = -1
        self.word_entity_combo = ("", "")  # Set a default value
        self.number_of_queries = None
        self.pattern_basis = self.word_entity_combo

    @staticmethod
    def find_word_entity_unique_class(data: List[str], labels: List[int], words: List[str], entities: List[str], threshold: float = 0.8) -> Dict[str, Union[int, List[int]]]:
        WordEntityMixLabeller.threshold = threshold
        WordEntityMixLabeller.word_list = words
        WordEntityMixLabeller.entity_list = entities

        entity_word_counts = defaultdict(lambda: defaultdict(int))

        # Generate all combinations of words and entities
        word_entity_combos = list(itertools.product(WordEntityMixLabeller.word_list, WordEntityMixLabeller.entity_list))

        for sentence, label in zip(data, labels):
            for word_entity_combo in word_entity_combos:
                word, entity = word_entity_combo
                # Check if the word is in the sentence and the entity can be detected in the sentence
                entities_in_sentence = WordEntityMixLabeller.entity_detector.detect_entities(sentence)
                entities_in_sentence = [item["dim"] for item in entities_in_sentence]
                if word in sentence.split() and entity in entities_in_sentence:
                    entity_word_counts[word_entity_combo][label] += 1

        for word_entity_combo, label_counts in entity_word_counts.items():
            max_count = max(label_counts.values())
            total_count = sum(label_counts.values())

            # number of relevant queries use to generate lf is the max_count
            WordEntityMixLabeller.number_of_queries_map[word_entity_combo] = max_count

            if max_count / total_count >= threshold:
                WordEntityMixLabeller.entity_word_label_map[word_entity_combo] = max(label_counts, key=label_counts.get)
            else:
                WordEntityMixLabeller.entity_word_label_map[word_entity_combo] = -1

        return WordEntityMixLabeller.entity_word_label_map

    def find_word_entity_label(self, word_entity_combo: str) -> int:
        self.label = WordEntityMixLabeller.entity_word_label_map[word_entity_combo]
        self.word_entity_combo = word_entity_combo

        # stores number of queries used to generate the labelling functions
        self.number_of_queries = WordEntityMixLabeller.number_of_queries_map[word_entity_combo]

        return self.label

    def apply(self, sentence: str) -> int:
        entities = WordEntityMixLabeller.entity_detector.detect_entities(sentence)
        entities = [item["dim"] for item in entities]
        words = sentence.split()
        word_entity_combos = list(filter(lambda x: x[0] in words and x[1] in entities, itertools.product(WordEntityMixLabeller.word_list, WordEntityMixLabeller.entity_list)))

        if WordEntityMixLabeller.threshold == 1.0:
            for word_entity_combo in word_entity_combos:
                if word_entity_combo == self.word_entity_combo:
                    return self.label
            return -1
        else:
            if word_entity_combos and len(word_entity_combos) == 1 and word_entity_combos[0] == self.word_entity_combo:
                return self.label
            else:
                return -1

    def apply_generalised(self, sentence: str) -> int: # previously _generalised
        entities = WordEntityMixLabeller.entity_detector.detect_entities(sentence)
        entities = [item["dim"] for item in entities]
        words = sentence.split()
        word_entity_combos = list(filter(lambda x: x[0] in words and x[1] in entities, itertools.product(WordEntityMixLabeller.word_list, WordEntityMixLabeller.entity_list)))

        if WordEntityMixLabeller.threshold == 1.0:
            for word_entity_combo in word_entity_combos:
                if word_entity_combo in WordEntityMixLabeller.entity_word_label_map:
                    self.label = WordEntityMixLabeller.entity_word_label_map[word_entity_combo]
                    self.word_entity_combo = word_entity_combo
            self.label = -1
            self.word_entity_combo = word_entity_combos
        else:
            if word_entity_combos and len(word_entity_combos) == 1 and word_entity_combos[0] in WordEntityMixLabeller.entity_word_label_map:
                self.label = WordEntityMixLabeller.entity_word_label_map[word_entity_combos[0]]
                self.word_entity_combo = word_entity_combos
            else:
                self.label = -1
                self.word_entity_combo = word_entity_combos
                
        return self.label
