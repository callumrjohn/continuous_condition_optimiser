from sklearn.gaussian_process import GaussianProcessRegressor
from src.models.basemodel import BaseModel

class GPRModel(BaseModel):
    """
    Gaussian Process regression model wrapper implementing the BaseModel interface.
    
    Provides a scikit-learn compatible interface for Gaussian Process regressors
    with uncertainty quantification and model persistence capabilities.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize GPRModel with specified hyperparameters.
        
        Args:
            **kwargs: Hyperparameters passed to GaussianProcessRegressor (e.g.,
                kernel, alpha, optimizer, n_restarts_optimizer, etc.)
        """
        self._build_model(**kwargs)

    def __str__(self):
        """Return string representation of the model."""
        return "GaussianProcessRegressor"

    @property
    def model_params(self):
        """
        Get a dictionary of current model hyperparameters.
        
        Returns:
            Dictionary with model parameter information
        """
        return {
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": self.model.epsilon,
            "gamma": self.model.gamma,
            "degree": self.model.degree,
            "coef0": self.model.coef0
        }
        
    def _build_model(self, **kwargs):
        """
        Initialize the underlying GaussianProcessRegressor model.
        
        Args:
            **kwargs: Hyperparameters for GaussianProcessRegressor
        """
        self.model = GaussianProcessRegressor(**kwargs)

    def clear_model(self):
        """Reset the model to a fresh, untrained state."""
        self._build_model()

    def train(self, X, y):
        """
        Train the Gaussian Process model on the provided data.
        
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