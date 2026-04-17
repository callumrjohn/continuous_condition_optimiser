from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class RANDOMModel(BaseEstimator, RegressorMixin):
    """
    Baseline random prediction model for regression.
    
    Generates random predictions uniformly distributed between the minimum and
    maximum target values observed in the training data. Used as a baseline to
    evaluate whether more sophisticated models provide improvement.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize RANDOMModel.
        
        Args:
            **kwargs: Dummy keyword arguments to match the interface of other models
        """
        pass

    def __str__(self):
        """Return string representation of the model."""
        return "RANDOMModel"

    def clear_model(self):
        """Reset the model (no-op for random model)."""
        pass

    def fit(self, X, y):
        """
        Fit the model by recording the range of target values.
        
        Args:
            X: Feature matrix (unused)
            y: Target values used to determine prediction range
        
        Returns:
            self
        """
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        return self
    
    def train(self, X, y):
        """
        Train the model (calls fit method).
        
        Args:
            X: Feature matrix (unused)
            y: Target values used to determine prediction range
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Generate random predictions uniformly distributed in the observed range.
        
        Args:
            X: Feature matrix of shape [n_samples, n_features] (only used for determining output size)
        
        Returns:
            Random predictions as numpy array of shape [n_samples,]
        """
        return np.random.uniform(self.y_min, self.y_max, size=len(X))

    def save(self, path):
        """
        Save the model to a file using joblib.
        
        Args:
            path: File path where the model should be saved
        """
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a previously saved model from file using joblib.
        
        Args:
            path: File path to the saved model
        """
        import joblib
        self.model = joblib.load(path)