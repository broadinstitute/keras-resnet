import keras_resnet.block
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
            y = keras_resnet.block.residual(block, filters=filters, repetitions=repetition, first=(index == 0))(y)

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
        block = keras_resnet.block.basic

        super(ResNet18, self).__init__(x, classes, block, [2, 2, 2, 2])


class ResNet34(ResNet):
    def __init__(self, x, classes):
        block = keras_resnet.block.basic

        super(ResNet34, self).__init__(x, classes, block, [3, 4, 6, 3])


class ResNet50(ResNet):
    def __init__(self, x, classes):
        block = keras_resnet.block.bottleneck

        super(ResNet50, self).__init__(x, classes, block, [3, 4, 6, 3])


class ResNet101(ResNet):
    def __init__(self, x, classes):
        block = keras_resnet.block.bottleneck

        super(ResNet101, self).__init__(x, classes, block, [3, 4, 23, 3])


class ResNet152(ResNet):
    def __init__(self, x, classes):
        block = keras_resnet.block.bottleneck

        super(ResNet152, self).__init__(x, classes, block, [3, 8, 36, 3])
