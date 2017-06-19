# -*- coding: utf-8 -*-

"""

keras_resnet.model
~~~~~~~~~~~~~~~~~~

This module implements a number of popular residual networks.

"""

import keras_resnet.block
import keras.layers
import keras.backend
import keras.models
import keras.regularizers


class ResNet(keras.models.Model):
    """

    A custom :class:`ResNet <ResNet>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.basic`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> blocks = [2, 2, 2, 2]
        >>> block = keras_resnet.basic
        >>> model = keras_resnet.ResNet(x, classes, blocks, block)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks, block):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(
            inputs)
        x = keras.layers.BatchNormalization(axis=axis)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(
            x)

        features = 64

        for j, iterations in enumerate(blocks):
            for k in range(iterations):
                if k == 0 and not j == 0:
                    strides = (2, 2)
                else:
                    strides = (1, 1)

                x = block(features, strides, j == 0 and k == 0)(x)

            features *= 2

        x = keras.layers.BatchNormalization(axis=axis)(x)
        x = keras.layers.Activation("relu")(x)

        shape = keras.backend.int_shape(x)

        if keras.backend.image_data_format() == "channels_last":
            pool_size = (shape[1], shape[2])
        else:
            pool_size = (shape[2], shape[3])

        x = keras.layers.AveragePooling2D(pool_size, strides=(1, 1))(x)

        super(ResNet, self).__init__(inputs, x)


class ResNet18(ResNet):
    """

    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> y = keras_resnet.ResNet18(x)
        >>> y = keras.layers.Dense(4096, activation="relu")(y)
        >>> y = keras.layers.Dropout(0.5)(y)
        >>> y = keras.layers.Dense(classes, activation="softmax")(y)
        >>> model = keras.models.Model(x, y)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        block = keras_resnet.block.basic

        super(ResNet18, self).__init__(inputs, [2, 2, 2, 2], block)


class ResNet34(ResNet):
    """

    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> model = keras_resnet.ResNet34(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        block = keras_resnet.block.basic

        super(ResNet34, self).__init__(inputs, [3, 4, 6, 3], block)


class ResNet50(ResNet):
    """

    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> model = keras_resnet.ResNet50(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        block = keras_resnet.block.bottleneck

        super(ResNet50, self).__init__(inputs, [3, 4, 6, 3], block)


class ResNet101(ResNet):
    """

    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> model = keras_resnet.ResNet101(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        block = keras_resnet.block.bottleneck

        super(ResNet101, self).__init__(inputs, [3, 4, 23, 3], block)


class ResNet152(ResNet):
    """

    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> model = keras_resnet.ResNet152(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        block = keras_resnet.block.bottleneck

        super(ResNet152, self).__init__(inputs, [3, 8, 36, 3], block)


class ResNet200(ResNet):
    """

    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage::
        >>> import keras_resnet
        >>> shape, classes = (32, 32, 3), 10
        >>> x = keras.layers.Input(shape)
        >>> model = keras_resnet.ResNet200(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        block = keras_resnet.block.bottleneck

        super(ResNet200, self).__init__(inputs, [3, 24, 36, 3], block)
