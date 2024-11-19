import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from program.labelling_functions.MLLabeller import MLLabeller


class RandomForestLabeller(MLLabeller):
    """
    Class for a labelling function that uses a random forest classifier.
    """

    def __init__(self, vectorizer: CountVectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english', strip_accents='unicode'), n_estimators: int = 128):
        """
        Random forest labelling function.
        """
        super().__init__("rfl", self.apply)
        self.label = -1
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.vectorizer = vectorizer
        self.dataset = None
        self.pattern_basis = self.dataset
        self.number_of_queries = None

    def fit(self, data, labels) -> None:

        # Check if the vectorizer has been fitted and transformed already
        if not hasattr(self.vectorizer, 'vocabulary_'):
            data = self.vectorizer.fit_transform(data)

        self.model.fit(data, labels)

    def tune_hyperparameters(self, data, labels):
        # Define the grid of hyperparameters to search
        hyperparameter_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Set up the grid search with 5-fold cross validation
        grid_cv = GridSearchCV(estimator=self.model, param_grid=hyperparameter_grid, cv=5, n_jobs=-1)

        # Fit the grid search to the data
        grid_cv.fit(data, labels)

        # Get the best parameters
        best_params = grid_cv.best_params_
        print("Best parameters: ", best_params)

        # Update the model with the best parameters
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