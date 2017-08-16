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

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet(inputs, blocks, block, include_top=True, classes=1000, *args, **kwargs):
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same"), name="conv1")(inputs)
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis), name="bn_conv1")(x)
    x = keras.layers.TimeDistributed(keras.layers.Activation("relu"), name="conv1_relu")(x)
    x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"), name="pool1")(x)

    features = 64

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(blocks[stage_id] > 6))(x)

        features *= 2

    if include_top:
        assert classes > 0

        x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D(), name="pool5")(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"), name="fc1000")(x)

    return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet18 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet18(inputs, blocks=[2, 2, 2, 2], include_top=True, classes=1000, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet34 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet34(inputs, blocks=[3, 4, 6, 3], include_top=True, classes=1000, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet50 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet50(inputs, blocks=[3, 4, 6, 3], include_top=True, classes=1000, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet101 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet101(inputs, blocks=[3, 4, 23, 3], include_top=True, classes=1000, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet152 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet152(inputs, blocks=[3, 8, 36, 3], include_top=True, classes=1000, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


"""

Constructs a time distributed `keras.models.Model` according to the ResNet200 specifications.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

:param blocks: the network’s residual architecture

:param include_top: if true, includes classification layers

:param classes: number of classes to classify (include_top must be true)

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
def TimeDistributedResNet200(inputs, blocks=[3, 24, 36, 3], include_top=True, classes=1000, *args, **kwargs):
    return TimeDistributedResNet(inputs, blocks, block=keras_resnet.blocks.time_distributed_bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)
