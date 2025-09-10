import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd

def build_model(input_shape):
    """
    Builds a simple sequential Keras model.

    Args:
        input_shape (int): The number of input features.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Trains the Keras model.

    Args:
        model (tf.keras.Model): The Keras model to train.
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training target.
        X_val (pd.DataFrame): The validation data.
        y_val (pd.Series): The validation target.

    Returns:
        tf.keras.Model: The trained Keras model.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping], verbose=0)
    return model

def generate_submission(model, test_df, passenger_ids):
    """
    Generates the submission file.

    Args:
        model (tf.keras.Model): The trained model.
        test_df (pd.DataFrame): The test data.
        passenger_ids (pd.Series): The passenger IDs for the submission file.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    preds = (model.predict(test_df) > 0.5).astype(int)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds.flatten()})
    return submission_df
