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

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks, block, include_top=True, classes=1000):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1")(inputs)
        x = keras.layers.BatchNormalization(axis=axis, name="bn_conv1")(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(features, stage_id, block_id, numerical_name=(blocks[stage_id] > 6))(x)

            features *= 2

        if include_top:
            assert classes > 0

            x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
            x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        super(ResNet, self).__init__(inputs, x)


class ResNet18(ResNet):
    """

    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet18(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        block = keras_resnet.blocks.basic_2d

        super(ResNet18, self).__init__(inputs, blocks, block, include_top=include_top, classes=classes)


class ResNet34(ResNet):
    """

    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet34(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        block = keras_resnet.blocks.basic_2d

        super(ResNet34, self).__init__(inputs, blocks, block, include_top=include_top, classes=classes)


class ResNet50(ResNet):
    """

    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet50(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet50, self).__init__(inputs, blocks, block, include_top=include_top, classes=classes)


class ResNet101(ResNet):
    """

    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet101(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet101, self).__init__(inputs, blocks, block, include_top=include_top, classes=classes)


class ResNet152(ResNet):
    """

    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet152(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet152, self).__init__(inputs, blocks, block, include_top=include_top, classes=classes)


class ResNet200(ResNet):
    """

    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: list of blocks to use

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet200(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        block = keras_resnet.blocks.bottleneck_2d

        super(ResNet200, self).__init__(inputs, blocks, block, include_top=include_top, classes=classes)
