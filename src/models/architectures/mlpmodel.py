import tensorflow as tf
from tensorflow.keras import layers, models

class MLPModel:
    """
    Multi-Layer Perceptron neural network model for regression.
    
    Implements a Keras sequential model with configurable hidden layers,
    dropout regularization, and early stopping to prevent overfitting.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize MLPModel with neural network architecture parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - hidden_layers: List of hidden layer sizes (default: [32, 16])
                - activation: Activation function (default: 'relu')
                - dropout_rate: Dropout rate for regularization (default: 0.4)
                - optimizer: Optimizer algorithm (default: 'adam')
                - learning_rate: Learning rate (default: 0.001)
                - loss: Loss function - 'mean_squared_error', 'huber', etc. (default: 'mse')
                - delta: Delta parameter for Huber loss (default: 1.0)
                - input_shape: Input feature dimension (default: (1,))
                - output_units: Output dimension (default: 1)
                - epochs: Training epochs (default: 100)
                - batch_size: Batch size (default: 8)
        """
        self.hidden_layers = kwargs.get("hidden_layers", [32, 16])
        self.activation = kwargs.get("activation", "relu")
        self.dropout_rate = kwargs.get("dropout_rate", 0.4)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.learning_rate = kwargs.get("learning_rate", 0.001)

        loss_name = kwargs.get("loss", "mean_squared_error")
        if loss_name == "huber":
            delta = kwargs.get("delta", 1.0)
            self.loss = tf.keras.losses.Huber(delta=delta)
        else:
            self.loss = loss_name
        
        self.input_shape = kwargs.get("input_shape", (1,))
        self.output_units = kwargs.get("output_units", 1)
        self.epochs = kwargs.get("epochs", 100)
        self.batch_size = kwargs.get("batch_size", 8)
        self._build_model()

    def __str__(self):
        """Return string representation of the model."""
        return "MultiLayerPerceptron"

    @property
    def model_params(self):
        """
        Get a dictionary of current model architecture and training parameters.
        
        Returns:
            Dictionary with all configuration parameters
        """
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
        """
        Build the neural network architecture with configured parameters.
        
        Creates a sequential model with:
        - Input layer with specified shape
        - Hidden layers with configurable sizes and activation
        - Dropout layers for regularization
        - Output layer with linear activation for regression
        """
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=self.input_shape))
        for units in self.hidden_layers:
            self.model.add(layers.Dense(units, activation=self.activation))
            self.model.add(layers.Dropout(self.dropout_rate))
        self.model.add(layers.Dense(self.output_units, activation='linear'))
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer_instance, loss=self.loss)

    def clear_model(self):
        """Reset the model to a fresh, uninitialized state."""
        self._build_model()

    def train(self, X, y):
        """
        Train the neural network model on the provided data.
        
        Uses early stopping to prevent overfitting and validation split
        for monitoring generalization performance.
        
        Args:
            X: Feature matrix of shape [n_samples, n_features]
            y: Target values of shape [n_samples,] or [n_samples, 1]
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=10, 
                                                    restore_best_weights=True)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks=[callback]) # reduce overfitting for small datasets

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix of shape [n_samples, n_features]
        
        Returns:
            Predicted values as numpy array of shape [n_samples,] or [n_samples, 1]
        """
        return self.model.predict(X)

    def save(self, path):
        """
        Save the trained model to a file.
        
        Args:
            path: File path where the model and weights should be saved
        """
        self.model.save(path)

    def load(self, path):
        """
        Load a previously saved model from file.
        
        Args:
            path: File path to the saved model
        """
        self.model = models.load_model(path)