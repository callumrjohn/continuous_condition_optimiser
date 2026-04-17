from sklearn.svm import SVR
from src.models.basemodel import BaseModel

class SVRModel(BaseModel):
    """
    Support Vector Machine regression model wrapper implementing the BaseModel interface.
    
    Provides a scikit-learn compatible interface for Support Vector regressors
    with model persistence capabilities (save/load via joblib).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize SVRModel with specified hyperparameters.
        
        Args:
            **kwargs: Hyperparameters passed to SVR (e.g., kernel, C, epsilon,
                gamma, degree, coef0, etc.)
        """
        self._build_model(**kwargs)

    def __str__(self):
        """Return string representation of the model."""
        return "SupportVectorRegressor"

    @property
    def model_params(self):
        """
        Get a dictionary of current model hyperparameters.
        
        Returns:
            Dictionary with keys: kernel, C, epsilon, gamma, degree, coef0
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
        Initialize the underlying SVR model.
        
        Args:
            **kwargs: Hyperparameters for SVR
        """
        self.model = SVR(**kwargs)

    def clear_model(self):
        """Reset the model to a fresh, untrained state."""
        self._build_model()

    def train(self, X, y):
        """
        Train the SVM model on the provided data.
        
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