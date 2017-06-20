# -*- coding: utf-8 -*-

"""

keras_resnet.models._time_distributed_2d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular time distributed two-dimensional residual networks.

"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.blocks


class TimeDistributedResNet(keras.models.Model):
    """
    
    A custom :class:`TimeDistributedResNet <TimeDistributedResNet>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    
    :param blocks: the network’s residual architecture

    :param block: a time distributed residual block (e.g. an instance of `keras_resnet.blocks.time_distributed_basic_2d`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> blocks = keras_resnet.blocks.time_distributed_basic_2d

        >>> y = keras_resnet.models.TimeDistributedResNet(x, classes, blocks, blocks)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs, blocks, block):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same"))(inputs)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(x)
        x = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(x)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"))(x)

        features = 64

        for j, iterations in enumerate(blocks):
            for k in range(iterations):
                if k == 0 and not j == 0:
                    strides = (2, 2)
                else:
                    strides = (1, 1)

                x = block(features, strides, j == 0 and k == 0)(x)

            features *= 2

        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(x)
        x = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(x)

        shape = keras.backend.int_shape(x)

        if keras.backend.image_data_format() == "channels_last":
            pool_size = (shape[1], shape[2])
        else:
            pool_size = (shape[2], shape[3])

        x = keras.layers.TimeDistributed(keras.layers.AveragePooling2D(pool_size, strides=(1, 1)))(x)

        super(TimeDistributedResNet, self).__init__(inputs, x)


class TimeDistributedResNet18(TimeDistributedResNet):
    """
    
    A :class:`TimeDistributedResNet18 <TimeDistributedResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.TimeDistributedResNet18(x)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs):
        blocks = [2, 2, 2, 2]

        block = keras_resnet.blocks.time_distributed_basic_2d

        super(TimeDistributedResNet18, self).__init__(inputs, blocks, block)


class TimeDistributedResNet34(TimeDistributedResNet):
    """
    
    A :class:`TimeDistributedResNet34 <TimeDistributedResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.TimeDistributedResNet34(x)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs):
        blocks = [3, 4, 6, 3]

        block = keras_resnet.blocks.time_distributed_basic_2d

        super(TimeDistributedResNet34, self).__init__(inputs, blocks, block)


class TimeDistributedResNet50(TimeDistributedResNet):
    """
    
    A :class:`TimeDistributedResNet50 <TimeDistributedResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.TimeDistributedResNet50(x)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs):
        blocks = [3, 4, 6, 3]

        block = keras_resnet.blocks.time_distributed_bottleneck_2d

        super(TimeDistributedResNet50, self).__init__(inputs, blocks, block)


class TimeDistributedResNet101(TimeDistributedResNet):
    """
    
    A :class:`TimeDistributedResNet101 <TimeDistributedResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.TimeDistributedResNet101(x)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs):
        blocks = [3, 4, 23, 3]

        block = keras_resnet.blocks.time_distributed_bottleneck_2d

        super(TimeDistributedResNet101, self).__init__(inputs, blocks, block)


class TimeDistributedResNet152(TimeDistributedResNet):
    """
    
    A :class:`TimeDistributedResNet152 <TimeDistributedResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.TimeDistributedResNet152(x)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs):
        blocks = [3, 8, 36, 3]

        block = keras_resnet.blocks.time_distributed_bottleneck_2d

        super(TimeDistributedResNet152, self).__init__(inputs, blocks, block)


class TimeDistributedResNet200(TimeDistributedResNet):
    """
    
    A :class:`TimeDistributedResNet200 <TimeDistributedResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> y = keras_resnet.models.TimeDistributedResNet200(x)

        >>> y = keras.layers.TimeDistributed(keras.layers.Flatten())(y.output)

        >>> y = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        >>> model = keras.models.Model(x, y)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    
    """

    def __init__(self, inputs):
        blocks = [3, 24, 36, 3]

        block = keras_resnet.blocks.time_distributed_bottleneck_2d

        super(TimeDistributedResNet200, self).__init__(inputs, blocks, block)
