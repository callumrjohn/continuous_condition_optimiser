import numpy as np
import pytest


#----------------- Test function for BaseModel abstract class ----------------
def test_basemodel_is_abstract():
    from src.models.basemodel import BaseModel

    try:
        BaseModel()
        assert False, "BaseModel should not be instantiable"
    except TypeError:
        assert True


#----------------- Test function for RANDOMModel ----------------
def test_random_model_train_predict():
    from src.models.architectures.randommodel import RANDOMModel

    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    X_test = np.array([[0.5], [1.5], [2.5]])

    model = RANDOMModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (3,)
    assert np.all(y_pred >= 10.0)
    assert np.all(y_pred <= 40.0)


#----------------- Test function for RFModel ----------------
def test_rfmodel_train_predict_and_params():
    from src.models.architectures.rfmodel import RFModel

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0]])
    X_test = np.array([[1.5], [2.5]])

    model = RFModel(n_estimators=10, random_state=0)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (2,)

    params = model.model_params
    assert "n_estimators" in params
    assert params["n_estimators"] == 10


#----------------- Test function for SVRModel ----------------
def test_svrmodel_train_predict_and_params():
    from src.models.architectures.svrmodel import SVRModel

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0]])
    X_test = np.array([[1.5], [2.5]])

    model = SVRModel(kernel="rbf", C=1.0, epsilon=0.1)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (2,)

    params = model.model_params
    assert params["kernel"] == "rbf"
    assert np.isclose(params["C"], 1.0)
    assert np.isclose(params["epsilon"], 0.1)


#----------------- Test function for GPRModel ----------------
def test_gprmodel_train_predict():
    from src.models.architectures.gprmodel import GPRModel

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0]])
    X_test = np.array([[1.5], [2.5]])

    model = GPRModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (2,)


#----------------- Test function for MLPModel ----------------
def test_mlpmodel_train_predict_and_params():
    pytest.importorskip("tensorflow")
    from src.models.architectures.mlpmodel import MLPModel

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    X_test = np.array([[1.5], [2.5]])

    model = MLPModel(hidden_layers=[8], epochs=1, batch_size=2, input_shape=(1,), output_units=1)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (2, 1)

    params = model.model_params
    assert params["hidden_layers"] == [8]
    assert params["epochs"] == 1
    assert params["batch_size"] == 2


#----------------- Test function for XGBModel ----------------
def test_xgbmodel_train_predict():
    pytest.importorskip("xgboost")
    from src.models.architectures.xgbmodel import XGBModel

    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    X_test = np.array([[1.5], [2.5]])

    model = XGBModel(n_estimators=5, max_depth=2, random_state=0)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (2,)
