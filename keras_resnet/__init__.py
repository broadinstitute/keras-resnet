import keras
import keras.datasets
import keras.layers
import keras.layers.convolutional
import keras.backend
import keras.layers.merge
import keras.layers.normalization
import keras.models
import keras.regularizers
import six

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


class ResNet(keras.models.Model):
    def __init__(self, x, classes, block, repetitions):
        if isinstance(block, six.string_types):
            block = globals().get(block)

            if not block:
                raise ValueError("Invalid {}".format(block))

        if keras.backend.image_data_format() == "channels_first":
            axis = 1
        else:
            axis = 3

        y = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), **parameters)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(y)

        filters = 64

        for index, repetition in enumerate(repetitions):
            y = residual(block, filters=filters, repetitions=repetition, first=(index == 0))(y)

            filters *= 2

        y = keras.layers.BatchNormalization(axis=axis)(y)
        y = keras.layers.Activation("relu")(y)

        block_shape = keras.backend.int_shape(y)

        y = keras.layers.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(y)

        y = keras.layers.Flatten()(y)

        y = keras.layers.Dense(units=classes, kernel_initializer="he_normal", activation="softmax")(y)

        super(ResNet, self).__init__(x, y)


class ResNet18(ResNet):
    def __init__(self, x, classes):
        super(ResNet18, self).__init__(x, classes, basic, [2, 2, 2, 2])


class ResNet34(ResNet):
    def __init__(self, x, classes):
        super(ResNet34, self).__init__(x, classes, basic, [3, 4, 6, 3])


class ResNet50(ResNet):
    def __init__(self, x, classes):
        super(ResNet50, self).__init__(x, classes, bottleneck, [3, 4, 6, 3])


class ResNet101(ResNet):
    def __init__(self, x, classes):
        super(ResNet101, self).__init__(x, classes, bottleneck, [3, 4, 23, 3])


class ResNet152(ResNet):
    def __init__(self, x, classes):
        super(ResNet152, self).__init__(x, classes, bottleneck, [3, 8, 36, 3])
