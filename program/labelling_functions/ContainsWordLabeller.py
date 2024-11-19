from program.labelling_functions.LabellingFunction import LabellingFunction


class ContainsWordLabeller(LabellingFunction):
    def __init__(self, words: list):
        super().__init__(f"cwl_{', '.join(words)}", self.apply)
        self.words = words
        self.label = -1
        self.number_of_queries = None
        self.pattern_basis = self.words


    def find_label(self, data, labels, intents, threshold: float = 0.8) -> int:
        """
        Generates the label to be used in this labelling function.

        :param data: Labelled documents in string format
        :param labels: Labels for the documents
        :param intents: Intents for the documents
        :param threshold: Threshold for the amount of noise acceptable.
        :return: Label of the labelling function
        """
        counts = {}
        for i in range(len(intents)):
            counts[i] = 0

        for sentence, label in zip(data, labels):
            all_in = True
            for word in self.words:
                if word not in sentence:
                    all_in = False
            if all_in:
                counts[label] += 1

        # Find the label with the most votes
        max_count = 0
        count_total = 0
        max_label = -1
        for label, count in counts.items():
            if count > max_count:
                max_count = count
                count_total += count
                max_label = label

        if max_label == -1:
            return -1

        # stores number of queries used to generate the labelling functions
        self.number_of_queries = max_count

        if max_count / count_total > threshold:
            self.label = max_label
            return max_label
        else:
            return -1

    def apply(self, sentence) -> int:
        """
        Returns matrix column of the labelling function's label on the data

        :param sentence: Documents to label in string format
        :return: Matrix column
        """
        all_in = True
        for word in self.words:
            if word not in sentence:
                all_in = False
        if all_in:
            return self.label
        return -1
