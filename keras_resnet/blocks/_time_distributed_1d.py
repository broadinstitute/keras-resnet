# -*- coding: utf-8 -*-

"""
keras_resnet.blocks._time_distributed_1d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements a number of popular time distributed one-dimensional
residual blocks.
"""

import warnings

import keras.layers
import keras.regularizers


def time_distributed_basic_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=(3, 3),
    strides=None,
    **kwargs
):
    """
    A time distributed one-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param strides: int representing the stride used in the shortcut and the
    first conv layer, default derives stride from block id

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.time_distributed_basic_1d(64)
    """
    if "freeze_bn" in kwargs:
        message = """

        The `freeze_bn` argument was depreciated in version 0.2.0 of 
        Keras-ResNet. It will be removed in version 0.3.0. 

        You can replace `freeze_bn=True` with:

                batch_normalization={"trainable": False}
        """

        warnings.warn(message)

    if "numerical_names" in kwargs:
        message = """

        The `numerical_names` argument was depreciated in version 0.2.0 of 
        Keras-ResNet. It will be removed in version 0.3.0. 
        """

        warnings.warn(message)

    if "stride" in kwargs:
        message = """

        The `stride` argument was renamed to `strides` in version 0.2.0 of 
        Keras-ResNet. It will be removed in version 0.3.0. 

        You can replace `stride=` with:

                strides=
        """

        warnings.warn(message)

        strides = kwargs["stride"]

    if "batch_normalization" in kwargs:
        batch_normalization_kwargs = kwargs["batch_normalization"]
    else:
        batch_normalization_kwargs = {}

    convolution_kwargs = {
        "kernel_initializer": "he_normal",
        "use_bias": False
    }

    if "convolution" in kwargs:
        convolution_kwargs = convolution_kwargs.update(kwargs["convolution"])

    if strides is None:
        if block != 0 or stage == 0:
            strides = 1
        else:
            strides = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and kwargs["numerical_names"]:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.TimeDistributed(
            keras.layers.ZeroPadding2D(
                padding=1
            ),
            name="padding{}{}_branch2a".format(stage_char, block_char)
        )(x)

        y = keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                **convolution_kwargs
            ),
            name="res{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(
                axis=axis,
                **batch_normalization_kwargs
            ),
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Activation(
                activation="relu"
            ),
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.ZeroPadding2D(
                padding=1
            ),
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                **convolution_kwargs
            ),
            name="res{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(
                axis=axis,
                **batch_normalization_kwargs
            ),
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = keras.layers.TimeDistributed(
                keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=(1, 1),
                    strides=strides,
                    **convolution_kwargs
                ),
                name="res{}{}_branch1".format(stage_char, block_char)
            )(x)

            shortcut = keras.layers.TimeDistributed(
                keras.layers.BatchNormalization(
                    axis=axis,
                    **batch_normalization_kwargs
                ),
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = keras.layers.TimeDistributed(
            keras.layers.Activation(
                activation="relu"
            ),
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f


def time_distributed_bottleneck_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=(3, 3),
    strides=None,
    **kwargs
):
    """

    A time distributed one-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param strides: int representing the stride used in the shortcut and the
    first conv layer, default derives stride from block id

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.time_distributed_bottleneck_1d(64)

    """
    if "freeze_bn" in kwargs:
        message = """

        The `freeze_bn` argument was depreciated in version 0.2.0 of 
        Keras-ResNet. It will be removed in version 0.3.0. 

        You can replace `freeze_bn=True` with:

                batch_normalization={"trainable": False}
        """

        warnings.warn(message)

    if "numerical_names" in kwargs:
        message = """

        The `numerical_names` argument was depreciated in version 0.2.0 of 
        Keras-ResNet. It will be removed in version 0.3.0. 
        """

        warnings.warn(message)

    if "stride" in kwargs:
        message = """

        The `stride` argument was renamed to `strides` in version 0.2.0 of 
        Keras-ResNet. It will be removed in version 0.3.0. 

        You can replace `stride=` with:

                strides=
        """

        warnings.warn(message)

        strides = kwargs["stride"]

    if "batch_normalization" in kwargs:
        batch_normalization_kwargs = kwargs["batch_normalization"]
    else:
        batch_normalization_kwargs = {}

    convolution_kwargs = {
        "kernel_initializer": "he_normal",
        "use_bias": False
    }

    if "convolution" in kwargs:
        convolution_kwargs = convolution_kwargs.update(kwargs["convolution"])

    if strides is None:
        if block != 0 or stage == 0:
            strides = 1
        else:
            strides = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and kwargs["numerical_names"]:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=(1, 1),
                strides=strides,
                **convolution_kwargs
            ),
            name="res{}{}_branch2a".format(stage_char, block_char)
        )(x)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(
                axis=axis,
                **batch_normalization_kwargs
            ),
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Activation(
                activation="relu"
            ),
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.ZeroPadding2D(
                padding=1
            ),
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                **convolution_kwargs
            ), name="res{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(
                axis=axis,
                **batch_normalization_kwargs
            ),
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Activation(
                activation="relu"
            ),
            name="res{}{}_branch2b_relu".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=filters * 4,
                kernel_size=(1, 1),
                **convolution_kwargs
            ), name="res{}{}_branch2c".format(stage_char, block_char)
        )(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(
                axis=axis,
                **batch_normalization_kwargs
            ),
            name="bn{}{}_branch2c".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = keras.layers.TimeDistributed(
                keras.layers.Conv1D(
                    filters=filters * 4,
                    kernel_size=(1, 1),
                    strides=strides,
                    **convolution_kwargs
                ),
                name="res{}{}_branch1".format(stage_char, block_char)
            )(x)

            shortcut = keras.layers.TimeDistributed(
                keras.layers.BatchNormalization(
                    axis=axis,
                    **batch_normalization_kwargs
                ),
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = keras.layers.TimeDistributed(
            keras.layers.Activation(
                activation="relu"
            ),
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f
