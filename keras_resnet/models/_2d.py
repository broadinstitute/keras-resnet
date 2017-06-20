# -*- coding: utf-8 -*-

"""

keras_resnet.models._2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional residual models.

"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.blocks


class ResNet(keras.models.Model):
    """

    A custom :class:`ResNet <ResNet>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual blocks (e.g. an instance of `keras_resnet.basic`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> y = keras_resnet.models.ResNet(x, classes, blocks, block)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks, block):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(inputs)
        x = keras.layers.BatchNormalization(axis=axis)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

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

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.ResNet18(x)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        blocks = [2, 2, 2, 2]

        block = keras_resnet.blocks.basic_2d

        super(ResNet18, self).__init__(inputs, blocks, block)


class ResNet34(ResNet):
    """

    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.ResNet34(x)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        blocks = [3, 4, 6, 3]

        block = keras_resnet.blocks.basic_2d

        super(ResNet34, self).__init__(inputs, blocks, block)


class ResNet50(ResNet):
    """

    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.ResNet50(x)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        blocks = [3, 4, 6, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet50, self).__init__(inputs, blocks, block)


class ResNet101(ResNet):
    """

    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.ResNet101(x)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        blocks = [3, 4, 23, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet101, self).__init__(inputs, blocks, block)


class ResNet152(ResNet):
    """

    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.ResNet152(x)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        blocks = [3, 8, 36, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet152, self).__init__(inputs, blocks, block)


class ResNet200(ResNet):
    """

    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.ResNet200(x)

        >>> y = keras.layers.Flatten()(y.output)

        >>> y = keras.layers.Dense(classes, activation="softmax")(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs):
        blocks = [3, 24, 36, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet200, self).__init__(inputs, blocks, block)
