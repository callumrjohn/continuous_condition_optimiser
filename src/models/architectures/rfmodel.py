from sklearn.ensemble import RandomForestRegressor
from src.models.basemodel import BaseModel

class RFModel(BaseModel):
    """
    Random Forest regression model wrapper implementing the BaseModel interface.
    
    Provides a scikit-learn compatible interface for Random Forest regressors
    with model persistence capabilities (save/load via joblib).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize RFModel with specified hyperparameters.
        
        Args:
            **kwargs: Hyperparameters passed to RandomForestRegressor (e.g.,
                n_estimators, max_depth, min_samples_split, etc.)
        """
        self._build_model(**kwargs)

    def __str__(self):
        """Return string representation of the model."""
        return "RandomForestRegressor"

    @property
    def model_params(self):
        """
        Get a dictionary of current model hyperparameters.
        
        Returns:
            Dictionary with keys: n_estimators, max_depth, min_samples_split,
            min_samples_leaf, max_features, bootstrap
        """
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "bootstrap": self.model.bootstrap
        }
        
    def _build_model(self, **kwargs):
        """
        Initialize the underlying RandomForestRegressor model.
        
        Args:
            **kwargs: Hyperparameters for RandomForestRegressor
        """
        self.model = RandomForestRegressor(**kwargs)

    def clear_model(self):
        """Reset the model to a fresh, untrained state."""
        self._build_model()

    def train(self, X, y):
        """
        Train the Random Forest model on the provided data.
        
        Args:
            X: Feature matrix of shape [n_samples, n_features]
            y: Target values of shape [n_samples,] or [n_samples, 1]
        """
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix of shape [n_samples, n_features]
        
        Returns:
            Predicted values as numpy array of shape [n_samples,]
        """
        return self.model.predict(X)

    def save(self, path):
        """
        Save the trained model to a file using joblib.
        
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