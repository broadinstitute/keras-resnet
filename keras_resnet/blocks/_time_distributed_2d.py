# -*- coding: utf-8 -*-

"""

keras_resnet.blocks._time_distributed_2d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements a number of popular time distributed two-dimensional residual blocks.

"""

import keras.layers
import keras.regularizers

parameters = {
    "kernel_initializer": "he_normal"
}


def time_distributed_basic_2d(filters, strides=(1, 1), first=False):
    """

    A time distributed two-dimensional basic block.

    :param filters: the output’s feature space

    :param strides: the convolution’s stride

    :param first: whether this is the first instance inside a residual blocks

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.time_distributed_basic_2d(64)

    """
    def f(x):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        y = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, (3, 3), strides=strides, padding="same", **parameters))(x)

        y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        y = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, (3, 3), padding="same", **parameters))(y)

        y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(y)
        y = _shortcut(x, y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        return y

    return f


def time_distributed_bottleneck_2d(filters, strides=(1, 1), first=False):
    """

    A time distributed bottleneck two-dimensional block.

    :param filters: the output’s feature space

    :param strides: the convolution’s stride

    :param first: whether this is the first instance inside a residual blocks

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.time_distributed_bottleneck_2d(64)

    """
    def f(x):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if first:
            y = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, (1, 1), strides=strides, padding="same", **parameters))(x)
        else:
            y = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, (3, 3), strides=strides, padding="same", **parameters))(x)

        y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        y = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, (3, 3), padding="same", **parameters))(y)

        y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        y = keras.layers.TimeDistributed(keras.layers.Conv2D(filters * 4, (1, 1), **parameters))(y)

        y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=axis))(y)
        y = _shortcut(x, y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        return y

    return f


def _shortcut(a, b):
    a_shape = keras.backend.int_shape(a)[1:]
    b_shape = keras.backend.int_shape(b)[1:]

    if keras.backend.image_data_format() == "channels_last":
        x = int(round(a_shape[1] / b_shape[1]))
        y = int(round(a_shape[2] / b_shape[2]))

        if x > 1 or y > 1 or not a_shape[3] == b_shape[3]:
            a = keras.layers.TimeDistributed(keras.layers.Conv2D(b_shape[3], (1, 1), strides=(x, y), padding="same", **parameters))(a)

            a = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=3))(a)
    else:
        x = int(round(a_shape[2] / b_shape[2]))
        y = int(round(a_shape[3] / b_shape[3]))

        if x > 1 or y > 1 or not a_shape[1] == b_shape[1]:
            a = keras.layers.TimeDistributed(keras.layers.Conv2D(b_shape[1], (1, 1), strides=(x, y), padding="same", **parameters))(a)

            a = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=1))(a)

    return keras.layers.add([a, b])
