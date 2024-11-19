from typing import Tuple
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm


class SemanticGrouper:
    """
    Find similar sentences to the training set in the testing set
    and return the most similar sentences above a certain threshold.
    """

    def __init__(self, training_x, training_y, testing, model, intent_dict: dict = None,
                 transformer: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialise the necesary variables for the class.

        :param training_x: Examples from the training set.
        :param training_y: Labels from the training set.
        :param testing: Unlabelled examples from the testing set.
        :param intent_dict: Intent dictionary.
        :param transformer: Transformer model to use.
        """
        self.transformer = transformer
        print("Loading model...")
        
        self.training_x = training_x
        self.training_y = training_y
        self.testing = testing
        self.intent_dict = intent_dict
        self.model = model

    def get_similar_sentences(self, threshold: float = 0.7) -> Tuple[list, list]:
        if self.intent_dict is None:
            raise Exception('Intent dictionary not provided.')

        # Preparing the labelled dictionary remains unchanged
        labelled_dict = {}
        for intent, label in self.intent_dict.items():
            labelled_dict[intent] = [self.training_x[i] for i in range(len(self.training_y)) if self.training_y[i] == label]

        print("Calculating embeddings for labelled set...")
        labelled_embeddings = {}
        for intent, examples in tqdm(labelled_dict.items()):
            # Process embeddings in batches; assumes model.encode() handles batch processing efficiently.
            labelled_embeddings[intent] = self.model.encode(examples, convert_to_tensor=True, batch_size=32)  # Adjust batch_size based on memory constraints

        print("Calculating embeddings for unlabelled set... please wait...")
        # Calculate testing set embeddings in a single batch; adjust batch_size as necessary
        testing_embeddings = self.model.encode(self.testing, convert_to_tensor=True, batch_size=32)

        print("Calculating similarities...")
        sentence_intent = {}
        for index, t_embedding in tqdm(enumerate(testing_embeddings)):
            test_sentence = self.testing[index]
            for intent, l_embeddings in labelled_embeddings.items():
                # Calculate cosine similarities in a vectorized manner
                cos_scores = util.cos_sim(t_embedding, l_embeddings)
                
                # Use top_k to only consider matches above the threshold, reducing computation
                top_scores = torch.where(cos_scores >= threshold, cos_scores, torch.tensor(0.0))
                max_score = torch.max(top_scores)
                
                if max_score > 0:  # If there's at least one score above the threshold
                    if test_sentence not in sentence_intent or max_score > sentence_intent[test_sentence][1]:
                        sentence_intent[test_sentence] = (intent, max_score)

        # Collecting results remains unchanged
        results_x, results_y = [], []
        for sentence, (intent, _) in sentence_intent.items():
            results_x.append(sentence)
            results_y.append(self.intent_dict[intent])

        return results_x, results_y


class TwoSentenceSimilarity:
    """
    Takes in two sentences and returns the similarity between them.
    """

    def __init__(self, transformer: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.transformer = transformer
        self.model = SentenceTransformer(self.transformer, device='cpu')

    def get_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Takes in two sentences and returns the similarity between them.

        :param sentence1: The first sentence.
        :param sentence2: The second sentence.
        :return: The similarity between the two sentences.
        """
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        cos_scores = util.cos_sim(embeddings[0], embeddings[1])
        return cos_scores[0][0]
