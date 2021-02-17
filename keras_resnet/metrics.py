import tensorflow.keras.metrics


def top_1_categorical_error(y_true, y_pred):
    return 1.0 - tensorflow.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 1)


def top_5_categorical_error(y_true, y_pred):
    return 1.0 - tensorflow.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 5)
