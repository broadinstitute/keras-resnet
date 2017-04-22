import keras
import keras.datasets
import keras.layers
import keras.layers.convolutional
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

kernel_regularizer = keras.regularizers.l2(1.e-4)

parameters = {
    "kernel_initializer": "he_normal",
    "kernel_regularizer": kernel_regularizer,
    "padding": "same"
}


def _bn_relu_conv(**kwargs):
    kwargs = kwargs.copy()

    kwargs.update(parameters)

    def f(x):
        y = keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(x)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(**kwargs)(y)

        return y

    return f


def basic(filters, strides=(1, 1), top=False):
    def f(x):
        if top:
            y = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, **parameters)(x)
        else:
            y = keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(x)
            y = keras.layers.Activation("relu")(y)
            y = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, **parameters)(y)

        y = keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(x)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), **parameters)(y)

        return shortcut(x, y)

    return f


def bottleneck(filters, strides=(1, 1), top=False):
    def f(x):
        if top:
            y = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, **parameters)(x)
        else:
            y = keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(x)
            y = keras.layers.Activation("relu")(y)
            y = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, **parameters)(y)

        y = keras.layers.normalization.BatchNormalization(axis=CHANNEL_AXIS)(y)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), **parameters)(y)

        y = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(y)

        return shortcut(x, y)

    return f


def residual(block, filters, repetitions, top=False):
    def f(x):
        for index in range(repetitions):
            strides = (2, 2) if index == 0 and not top else (1, 1)

            x = block(filters, strides, (top and index == 0))(x)

        return x

    return f


def shortcut(a, b):
    input_shape = keras.backend.int_shape(a)

    residual_shape = keras.backend.int_shape(b)

    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))

    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        kernel_regularizer = keras.regularizers.l2(0.0001)

        a = keras.layers.Conv2D(filters=residual_shape[CHANNEL_AXIS], kernel_size=(1, 1), strides=(stride_width, stride_height), kernel_initializer="he_normal",
                                kernel_regularizer=kernel_regularizer)(a)

    return keras.layers.add([a, b])


class ResNet(keras.models.Model):
    def __init__(self, x, classes, block, repetitions):
        if isinstance(block, six.string_types):
            block = globals().get(block)

            if not block:
                raise ValueError("Invalid {}".format(block))

        y = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), **parameters)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(y)

        filters = 64

        for index, repetition in enumerate(repetitions):
            y = residual(block, filters=filters, repetitions=repetition, top=(index == 0))(y)

            filters *= 2

        y = keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(y)
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
