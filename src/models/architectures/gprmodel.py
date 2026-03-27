from sklearn.gaussian_process import GaussianProcessRegressor
from src.models.basemodel import BaseModel

class GPRModel(BaseModel):
    def __init__(self, **kwargs):
        self._build_model(**kwargs)

    def __str__(self):
        return "GaussianProcessRegressor"

    @property
    def model_params(self):
        return {
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": self.model.epsilon,
            "gamma": self.model.gamma,
            "degree": self.model.degree,
            "coef0": self.model.coef0
        }
        
    def _build_model(self, **kwargs):
        self.model = GaussianProcessRegressor(**kwargs)

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