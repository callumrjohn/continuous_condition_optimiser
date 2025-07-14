from xgboost import XGBRegressor
from src.models.basemodel import BaseModel

class XGBModel(BaseModel):
    def __init__(self, **kwargs):
        self._build_model(**kwargs)

    @property
    def model_params(self):
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