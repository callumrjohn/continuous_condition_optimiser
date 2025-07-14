import tensorflow as tf
from tensorflow.keras import layers, models

class MLPModel:
    def __init__(self, **kwargs):
        self.hidden_layers = kwargs.get("hidden_layers", [32, 16])
        self.activation = kwargs.get("activation", "relu")
        self.dropout_rate = kwargs.get("dropout_rate", 0.4)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.loss = kwargs.get("loss", "mean_squared_error")
        self.input_shape = kwargs.get("input_shape", (1,))
        self.output_units = kwargs.get("output_units", 1)
        self.epochs = kwargs.get("epochs", 100)
        self.batch_size = kwargs.get("batch_size", 8)
        self._build_model()

    @property
    def model_params(self):
        return {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "input_shape": self.input_shape,
            "output_units": self.output_units,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

    def _build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=self.input_shape))
        for units in self.hidden_layers:
            self.model.add(layers.Dense(units, activation=self.activation))
            self.model.add(layers.Dropout(self.dropout_rate))
        self.model.add(layers.Dense(self.output_units, activation='linear'))
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer_instance, loss=self.loss)

    def clear_model(self):
        self._build_model()

    def train(self, X, y):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=10, 
                                                    restore_best_weights=True)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks=[callback]) # reduce overfitting for small datasets

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = models.load_model(path)