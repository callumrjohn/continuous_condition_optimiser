from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class RANDOMModel(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs): # Dummy kwargs to match the interface
        pass

    def __str__(self):
        return "RANDOMModel"

    def clear_model(self):
        pass

    def fit(self, X, y):
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        return self
    
    def train(self, X, y):
        self.fit(X, y)

    def predict(self, X):
        return np.random.uniform(self.y_min, self.y_max, size=len(X))

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)