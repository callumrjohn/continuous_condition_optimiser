from xgboost import XGBRegressor
from src.models.basemodel import BaseModel

class XGBModel(BaseModel):
    """
    XGBoost regression model wrapper implementing the BaseModel interface.
    
    Provides a scikit-learn compatible interface for XGBoost gradient boosting
    models with model persistence capabilities (save/load).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize XGBModel with specified hyperparameters.
        
        Args:
            **kwargs: Hyperparameters passed to XGBRegressor (e.g., n_estimators,
                max_depth, learning_rate, subsample, etc.)
        """
        self._build_model(**kwargs)

    def __str__(self):
        """Return string representation of the model."""
        return "XGBoostRegressor"

    @property
    def model_params(self):
        """
        Get a dictionary of current model hyperparameters.
        
        Returns:
            Dictionary with keys: n_estimators, learning_rate, max_depth,
            subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
        """
        return {
            "n_estimators": self.model.get_booster().best_ntree_limit,
            "learning_rate": self.model.get_params().get("learning_rate", 0.1),
            "max_depth": self.model.get_params().get("max_depth", 3),
            "subsample": self.model.get_params().get("subsample", 1.0),
            "colsample_bytree": self.model.get_params().get("colsample_bytree", 1.0),
            "gamma": self.model.get_params().get("gamma", 0),
            "reg_alpha": self.model.get_params().get("reg_alpha", 0),
            "reg_lambda": self.model.get_params().get("reg_lambda", 1)
        }
        
    def _build_model(self, **kwargs):
        """
        Initialize the underlying XGBRegressor model.
        
        Args:
            **kwargs: Hyperparameters for XGBRegressor
        """
        self.model = XGBRegressor(**kwargs)

    def clear_model(self):
        """Reset the model to a fresh, untrained state."""
        self._build_model()

    def train(self, X, y):
        """
        Train the XGBoost model on the provided data.
        
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
        Save the trained model to a file.
        
        Args:
            path: File path where the model should be saved
        """
        self.model.save_model(path)

    def load(self, path):
        """
        Load a previously saved model from file.
        
        Args:
            path: File path to the saved model
        """
        self.model.load_model(path)