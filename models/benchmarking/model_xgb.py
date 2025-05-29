from xgboost import XGBRegressor
from models.basemodel import BaseModel
class XGBModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)