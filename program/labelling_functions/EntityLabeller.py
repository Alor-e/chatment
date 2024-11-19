import sys
import os
import asyncio

# Get the path to the root project directory
root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(root_directory)

# Add the root directory to the Python path
sys.path.append(root_directory)

from collections import defaultdict
from typing import List, Dict, Union
from program.labelling_functions.LabellingFunction import LabellingFunction
from data.entity import EntityDetector


class EntityLabeller(LabellingFunction):
    """
    The EntityLabeller class is a subclass of the LabellingFunction class. It is used to label sentences based on the entities they contain.

    The class uses an EntityDetector to identify entities in each sentence. It then determines the label for each entity based on the labels of the sentences in which the entity appears. The label is determined based on a threshold, which specifies how unique an entity must be to a label to be considered 'unique'.

    Attributes:
        label (int): The label assigned by this labelling function. Initialized to -1.
        entity_detector (EntityDetector): An instance of the EntityDetector class used to detect entities in sentences.
        entity_label_map (dict): A dictionary mapping entities to their unique or almost unique labels.

    Methods:
        find_entity_unique_class(data: List[str], labels: List[int], intents: List[str], threshold: float = 0.8) -> Dict[str, Union[int, List[int]]]:
            Determines the unique or almost unique label for each entity in the data based on the specified threshold.

        apply(sentence: str) -> int:
            Returns the label for a sentence based on the entities it contains. If the sentence contains multiple entities, it returns -1.
    """

    entity_label_map = {}
    entity_detector = EntityDetector()
    threshold: float = 0.8
    number_of_queries_map = {}

    def __init__(self, key):
        super().__init__(f"entl{', '.join(key)}", self.apply)
        self.label = -1
        self.entity = ""
        self.number_of_queries = None
        self.pattern_basis = self.entity

    @staticmethod
    def find_entity_unique_class(data: List[str], labels: List[int], model_path: str, threshold: float = 0.8) -> Dict[str, Union[int, List[int]]]:
        """
        Generates the label to be used in this labelling function.

        :param data: Labelled documents in string format
        :param labels: Labels for the documents
        :param threshold: This value determines how unique an entity must be to a label to be considered 'unique'. A threshold of 1.0 requires an entity to appear only with a single label, while a lower threshold allows for some overlap with other labels.
        :return: Dictionary mapping entities to their unique or almost unique labels
        """
        # setting threshold
        EntityLabeller.threshold = threshold

        # Initialize a nested dictionary to count the occurrences of each entity for each label
        entity_counts = defaultdict(lambda: defaultdict(int))

        # Iterate over the data and labels in parallel
        for sentence, label in zip(data, labels):
            # Detect entities in the sentence
            entities = EntityLabeller.entity_detector.detect_entities(sentence)
            # print(sentence, entities) # to add log
            entities = [item["dim"] for item in entities]
            # Iterate over the detected entities
            for entity in entities:
                # Increment the count for the current entity and label
                entity_counts[entity][label] += 1

        rasa_entities_list = asyncio.run(EntityDetector.detect_entities_with_rasa(model_path, data, labels))
        print(rasa_entities_list)

        for item in rasa_entities_list:
            print(item)
            entity_counts[item["dim"]][item["label"]] += 1

        print("ENTITIY COUNTS")
        print(entity_counts)

        # Iterate over the entity counts
        for entity, label_counts in entity_counts.items():
            # Calculate the maximum count and the total count
            max_count = max(label_counts.values())
            total_count = sum(label_counts.values())

            # number of relevant queries use to generate lf is the max_count
            EntityLabeller.number_of_queries_map[entity] = max_count

            # If the ratio of the maximum count to the total count is greater than or equal to the threshold
            if max_count / total_count >= threshold:
                # Map the entity to the label with the maximum count
                EntityLabeller.entity_label_map[entity] = max(label_counts, key=label_counts.get)
            else:
                # Otherwise, map the entity to -1
                EntityLabeller.entity_label_map[entity] = -1 # to log out entity map

        return EntityLabeller.entity_label_map
    
    def find_entity_label(self, entity: str) -> int:
        """
        Returns the labelling of a given entity

        :param sentence: entity name
        :return: int
        """
        self.label = EntityLabeller.entity_label_map[entity]
        self.entity = entity

        # stores number of queries used to generate the labelling functions
        self.number_of_queries = EntityLabeller.number_of_queries_map[entity]

        return self.label

    def apply(self, sentence: str) -> int:
        """
        Returns matrix column of the labelling function's label on the data

        :param sentence: Documents to label in string format
        :return: int
        """
        # Detect entities in the sentence
        entities = EntityLabeller.entity_detector.detect_entities(sentence)
        entities = [item["dim"] for item in entities]

        # If the threshold is exactly 1
        if EntityLabeller.threshold == 1.0:
            # Iterate over the detected entities
            for entity in entities:
                # If the entity is the correct one for the labelling function
                if entity == self.entity:
                    return self.label
            # If none of the entities are in the map, return -1
            return -1
        else:
            # If exactly one entity is detected and it is in the entity-label map
            if entities and len(entities) == 1 and entities[0] == self.entity:
                # Return the corresponding label
                return self.label
            else:
                # Otherwise, return -1
                return -1
    
    def apply_generalised(self, sentence: str) -> int:
        """
        Returns matrix column of the labelling function's label on the data

        :param sentence: Documents to label in string format
        :return: Matrix column
        """
        # Detect entities in the sentence
        entities = EntityLabeller.entity_detector.detect_entities(sentence)
        entities = [item["dim"] for item in entities]
        # If the threshold is exactly 1
        if EntityLabeller.threshold == 1.0:
            # Iterate over the detected entities
            for entity in entities:
                # If the entity is in the entity-label map
                if entity in EntityLabeller.entity_label_map:
                    # Return the corresponding label and entity
                    self.label = EntityLabeller.entity_label_map[entity]
                    self.entity = entity
            # If none of the entities are in the map, return -1
            self.label = -1
            self.entity = entities
        else:
            # If exactly one entity is detected and it is in the entity-label map
            if entities and len(entities) == 1 and entities[0] in EntityLabeller.entity_label_map:

                # Return the corresponding label and entity
                self.label = EntityLabeller.entity_label_map[entities[0]]
                self.entity = entities
            else:
                # Otherwise, return -1
                self.label = -1
                self.entity = entities # entities[0] if len(entities) == 1 else entities
        
        return self.label


if __name__ == "__main__":
    # Initialize a list of sentences (data) and their corresponding labels
    data = ['Please tell me who removed My.exe, read.txt, and picture.jpeg files and also issue 449 in last week?',
                 'Who are the assignees on pulL Request with Number 456?',
                 'For this repo, show me the the number of Open Issues.',
                 'who opened issue 6744?']
    labels = [1, 2, 3, 4]

    # Initialize an instance of the EntityLabeller class
    entity_labeller = EntityLabeller()

    # Call the find_entity_unique_class method with the data, labels, intents, and a threshold of 0.8
    EntityLabeller.entity_label_map = entity_labeller.find_entity_unique_class(data, labels)
    print("Entity-Label Map:", EntityLabeller.entity_label_map)

    # Call the apply method with a sentence
    label = entity_labeller.apply("today is Monday the 25th")
    print("Label for 'today is Monday the 25th':", label)

    label = entity_labeller.apply("Find attached aa.zip")
    print("Label for 'Find attached aa.zip':", label)

    label = entity_labeller.apply("The event happened yesterday the 18th and the file is rr.mp4")
    print("Label for 'The event happened yesterday the 18th and the file is rr.mp4':", label)

    print(entity_labeller.entity)
    print(entity_labeller.label)
