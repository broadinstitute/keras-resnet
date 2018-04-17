# -*- coding: utf-8 -*-

"""
keras_resnet.models._feature_pyramid_2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional feature pyramids.
"""

import keras.backend

import keras_resnet.layers
import keras_resnet.models


def fpn(inputs, blocks, block, preamble=None, **kwargs):
    if "batch_normalization" in kwargs:
        batch_normalization_kwargs = kwargs["batch_normalization"]
    else:
        batch_normalization_kwargs = {}

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if "numerical_names" not in kwargs:
        numerical_names = [True] * len(blocks)

    x = keras.layers.ZeroPadding2D(name="padding_conv1", padding=3)(inputs)

    if preamble:
        x = preamble()(x)
    else:
        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            name="conv1",
            strides=(2, 2)
        )(x)

    x = keras.layers.BatchNormalization(
        axis=axis,
        name="bn_conv1",
        **batch_normalization_kwargs
    )(x)

    x = keras.layers.Activation(activation="relu", name="conv1_relu")(x)

    x = keras.layers.MaxPooling2D(
        name="pool1",
        padding="same",
        pool_size=(3, 3),
        strides=(2, 2)
    )(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            block_kargs = {}

            if "numerical_names" in kwargs:
                numerical_names = block_id > 0 and kwargs["numerical_names"][
                    stage_id]

            block_kargs.update({
                "numerical_names": numerical_names
            })

            x = block(
                block=block_id,
                filters=features,
                stage=stage_id,
                **block_kargs
            )(x)

        features *= 2

        outputs.append(x)

    kwargs.pop("numerical_names", None)

    pyramid_filters = 256

    c2, c3, c4, c5 = outputs

    pyramid_5 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        name="c5_reduced"
    )(c5)

    upsampled_p5 = keras_resnet.layers.Upsample(
        name="p5_upsampled"
    )([pyramid_5, c4])

    pyramid_5 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        name="p5"
    )(pyramid_5)

    pyramid_4 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        name="c4_reduced"
    )(c4)

    pyramid_4 = keras.layers.Add(
        name="p4_merged"
    )([upsampled_p5, pyramid_4])

    upsampled_p4 = keras_resnet.layers.Upsample(
        name="p4_upsampled"
    )([pyramid_4, c3])

    pyramid_4 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        name="p4"
    )(pyramid_4)

    pyramid_3 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        name="c3_reduced"
    )(c3)

    pyramid_3 = keras.layers.Add(
        name="p3_merged"
    )([upsampled_p4, pyramid_3])

    pyramid_3 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        name="p3"
    )(pyramid_3)

    pyramid_6 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        name="p6"
    )(c5)

    pyramid_7 = keras.layers.Activation(
        activation="relu",
        name="C6_relu"
    )(pyramid_6)

    pyramid_7 = keras.layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        name="p7"
    )(pyramid_7)

    return [pyramid_3, pyramid_4, pyramid_5, pyramid_6, pyramid_7]
