from sklearn.svm import SVR
from src.models.basemodel import BaseModel

class SVRModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = SVR(**kwargs)

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