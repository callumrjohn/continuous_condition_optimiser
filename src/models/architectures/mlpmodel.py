import tensorflow as tf
from tensorflow.keras import layers, models

class MLPModel:
    def __init__(self, input_shape, output_units, hidden_layers=[64, 32], activation='relu'):
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=input_shape))
        
        for units in hidden_layers:
            self.model.add(layers.Dense(units, activation=activation))
        
        self.model.add(layers.Dense(output_units, activation='linear'))  # Output layer for regression

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = models.load_model(path)