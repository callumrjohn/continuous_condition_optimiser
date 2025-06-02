from sklearn.ensemble import RandomForestRegressor
from src.models.basemodel import BaseModel

class RFModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)