import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Load dataset
    if cfg.training.dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        if len(cfg.model.input_shape) == 1:  # For simple NN
            x_train = x_train.reshape(-1, 784)
            x_test = x_test.reshape(-1, 784)
        else:  # For CNN
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
    else:
        raise ValueError("Unsupported dataset")

    # Initialize model
    model = Sequential()
    if cfg.model._target_ == "simple_nn":
        model.add(Dense(cfg.model.hidden_units[0], activation=cfg.model.activation, input_shape=cfg.model.input_shape))
        for units in cfg.model.hidden_units[1:]:
            model.add(Dense(units, activation=cfg.model.activation))
        model.add(Dense(cfg.model.output_units, activation='softmax'))
    elif cfg.model._target_ == "cnn":
        for layer in cfg.model.conv_layers:
            model.add(Conv2D(filters=layer.filters, kernel_size=layer.kernel_size, activation=layer.activation, input_shape=cfg.model.input_shape))
        model.add(Flatten())
        for units in cfg.model.dense_units:
            model.add(Dense(units, activation=cfg.model.activation))
        model.add(Dense(cfg.model.output_units, activation='softmax'))
    else:
        raise ValueError("Unsupported model type")

    # Compile model
    model.compile(optimizer=Adam(learning_rate=cfg.training.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=cfg.training.epochs, batch_size=cfg.training.batch_size, validation_data=(x_test, y_test))

if __name__ == "__main__":
    train()

