from sklearn.ensemble import RandomForestRegressor
from src.models.basemodel import BaseModel

class RFModel(BaseModel):
    def __init__(self, **kwargs):
        self._build_model(**kwargs)

    def __str__(self):
        return "RandomForestRegressor"

    @property
    def model_params(self):
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "bootstrap": self.model.bootstrap
        }
        
    def _build_model(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def clear_model(self):
        self._build_model()

    def train(self, X, y):
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)