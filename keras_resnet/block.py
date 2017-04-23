import keras
import keras.datasets
import keras.layers
import keras.layers.convolutional
import keras.backend
import keras.layers.merge
import keras.layers.normalization
import keras.models
import keras.regularizers

if keras.backend.image_dim_ordering() == "tf":
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3

parameters = {
    "kernel_initializer": "he_normal",
    "kernel_regularizer": keras.regularizers.l2(1.e-4),
    "padding": "same"
}


def convolution(**kwargs):
    kwargs = kwargs.copy()

    kwargs.update(parameters)

    def f(x):
        if keras.backend.image_data_format() == "channels_first":
            axis = 1
        else:
            axis = 3

        y = keras.layers.BatchNormalization(axis=axis)(x)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(**kwargs)(y)

        return y

    return f


def basic(filters, strides=(1, 1), first=False):
    def f(x):
        if keras.backend.image_data_format() == "channels_first":
            axis = 1
        else:
            axis = 3

        if first:
            y = keras.layers.Conv2D(filters, (3, 3), strides=strides, **parameters)(x)
        else:
            y = keras.layers.BatchNormalization(axis=axis)(x)
            y = keras.layers.Activation("relu")(y)
            y = keras.layers.Conv2D(filters, (3, 3), strides=strides, **parameters)(y)

        y = keras.layers.BatchNormalization(axis=axis)(y)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(filters, (3, 3), **parameters)(y)

        return shortcut(x, y)

    return f


def bottleneck(filters, strides=(1, 1), first=False):
    def f(x):
        if keras.backend.image_data_format() == "channels_first":
            axis = 1
        else:
            axis = 3

        if first:
            y = keras.layers.Conv2D(filters, (1, 1), strides=strides, **parameters)(x)
        else:
            y = keras.layers.BatchNormalization(axis=axis)(x)
            y = keras.layers.Activation("relu")(y)
            y = keras.layers.Conv2D(filters, (3, 3), strides=strides, **parameters)(y)

        y = keras.layers.normalization.BatchNormalization(axis=axis)(y)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(filters, (3, 3), **parameters)(y)

        y = convolution(filters=filters * 4, kernel_size=(1, 1))(y)

        return shortcut(x, y)

    return f


def residual(block, filters, repetitions, first=False):
    def f(x):
        for index in range(repetitions):
            strides = (2, 2) if index == 0 and not first else (1, 1)

            x = block(filters, strides, (first and index == 0))(x)

        return x

    return f


def shortcut(a, b):
    a_shape = keras.backend.int_shape(a)
    b_shape = keras.backend.int_shape(b)

    x = int(round(a_shape[ROW_AXIS] / b_shape[ROW_AXIS]))
    y = int(round(a_shape[COL_AXIS] / b_shape[COL_AXIS]))

    if x > 1 or y > 1 or not a_shape[CHANNEL_AXIS] == b_shape[CHANNEL_AXIS]:
        shortcut_parameters = {
            "kernel_initializer": "he_normal",
            "kernel_regularizer": keras.regularizers.l2(0.0001),
            "strides": (x, y)
        }

        a = keras.layers.Conv2D(b_shape[CHANNEL_AXIS], (1, 1), **shortcut_parameters)(a)

    return keras.layers.add([a, b])
