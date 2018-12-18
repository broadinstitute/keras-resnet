# -*- coding: utf-8 -*-

"""
keras_resnet.models._feature_pyramid_2d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional feature pyramid networks (FPNs).
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.blocks
import keras_resnet.layers


class FPN2D(keras.Model):
    def __init__(
            self,
            inputs,
            blocks,
            block,
            freeze_bn=True,
            numerical_names=None,
            *args,
            **kwargs
    ):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(inputs)
        x = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )(x)

            features *= 2

            outputs.append(x)

        c2, c3, c4, c5 = outputs

        pyramid_5 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c5_reduced"
        )(c5)

        upsampled_p5 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p5_upsampled",
            size=(2, 2)
        )(pyramid_5)

        pyramid_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c4_reduced"
        )(c4)

        pyramid_4 = keras.layers.Add(
            name="p4_merged"
        )([upsampled_p5, pyramid_4])

        upsampled_p4 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p4_upsampled",
            size=(2, 2)
        )(pyramid_4)

        pyramid_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p4"
        )(pyramid_4)

        pyramid_3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c3_reduced"
        )(c3)

        pyramid_3 = keras.layers.Add(
            name="p3_merged"
        )([upsampled_p4, pyramid_3])

        upsampled_p3 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p3_upsampled",
            size=(2, 2)
        )(pyramid_3)

        pyramid_3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p3"
        )(pyramid_3)

        pyramid_2 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c2_reduced"
        )(c2)

        pyramid_2 = keras.layers.Add(
            name="p2_merged"
        )([upsampled_p3, pyramid_2])

        pyramid_2 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p2"
        )(pyramid_2)

        pyramid_6 = keras.layers.MaxPooling2D(strides=2, name="p6")(pyramid_5)

        outputs = [
            pyramid_2,
            pyramid_3,
            pyramid_4,
            pyramid_5,
            pyramid_6
        ]

        super(FPN2D, self).__init__(
            inputs=inputs,
            outputs=outputs,
            *args,
            **kwargs
        )


class FPN2D50(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(FPN2D50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_2d,
            *args,
            **kwargs
        )


class FPN2D18(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        super(FPN2D18, self).__init__(
            inputs,
            blocks,
            block=keras_resnet.blocks.basic_2d,
            *args,
            **kwargs
        )


class FPN2D34(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(FPN2D34, self).__init__(
            inputs,
            blocks,
            block=keras_resnet.blocks.basic_2d,
            *args,
            **kwargs
        )


class FPN2D101(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(FPN2D101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_2d,
            *args,
            **kwargs
        )


class FPN2D152(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(FPN2D152, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_2d,
            *args,
            **kwargs
        )


class FPN2D200(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(FPN2D200, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_2d,
            *args,
            **kwargs
        )
