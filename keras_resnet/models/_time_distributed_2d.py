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


"""

Constructs a time distributed `keras.models.Model` object using the given block count.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param block: a time distributed residual block (e.g. an instance of `keras_resnet.blocks.time_distributed_basic_2d`)

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet(inputs, blocks, block, *args, **kwargs):
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

    return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet18 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet18(inputs, blocks=[2, 2, 2, 2], *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_basic_2d, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet34 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet34(inputs, blocks=[3, 4, 6, 3], *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_basic_2d, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet50 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet50(inputs, blocks=[3, 4, 6, 3], *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet101 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet101(inputs, blocks=[3, 4, 23, 3], *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet152 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet152(inputs, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet200 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:return model: Time distributed ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

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
def TimeDistributedResNet200(inputs, blocks=[3, 24, 36, 3], *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, *args, **kwargs)
