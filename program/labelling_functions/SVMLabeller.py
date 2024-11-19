
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from program.labelling_functions.MLLabeller import MLLabeller


class SVMLabeller(MLLabeller):
    """
    Class for a labelling function that uses a Support Vector Machine.
    """

    def __init__(self, vectorizer: CountVectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode'), kernel: str = 'rbf'):
        super().__init__("svm", self.apply)
        self.label = -1
        self.model = SVC(kernel=kernel, probability=True)
        self.vectorizer = vectorizer
        self.dataset = None
        self.pattern_basis = self.dataset
        self.number_of_queries = None

    def fit(self, data, labels) -> None:

        if not hasattr(self.vectorizer, 'vocabulary_'):
            data = self.vectorizer.fit_transform(data)

        self.model.fit(data, labels)

    def tune_hyperparameters(self, data, labels):
        hyperparameter_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        grid_cv = GridSearchCV(estimator=self.model, param_grid=hyperparameter_grid, cv=5, n_jobs=-1)
        grid_cv.fit(data, labels)

        best_params = grid_cv.best_params_
        print("Best parameters: ", best_params)

        self.model.set_params(**best_params)

    def apply(self, sentence) -> int:
        transformed_sentence = self.vectorizer.transform([sentence])
        
        # Get the predicted probabilities
        prediction_proba = self.model.predict_proba(transformed_sentence)
        
        # Find the maximum predicted probability
        max_prob = np.max(prediction_proba)
        
        # If max_prob is greater than 0.5, make a prediction, otherwise abstain (return -1)
        if max_prob > 0.5:
            prediction = self.model.predict(transformed_sentence)
            self.label = int(prediction[0])
            return self.label
        else:
            self.label = -1  # Abstain
            return self.label