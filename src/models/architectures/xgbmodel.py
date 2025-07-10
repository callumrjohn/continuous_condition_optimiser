from xgboost import XGBRegressor
from src.models.basemodel import BaseModel

class XGBModel(BaseModel):
    def __init__(self, **kwargs):
        self._build_model(**kwargs)
        
    def _build_model(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def clear_model(self):
        self._build_model()

    def train(self, X, y):
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)