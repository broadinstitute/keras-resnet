# -*- coding: utf-8 -*-

"""
keras_resnet.blocks._1d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements a number of popular one-dimensional residual blocks.
"""

from turtle import st
import keras.layers
import keras.regularizers

import keras_resnet.layers

parameters = {
    "kernel_initializer": "he_normal"
}

class Basic1D(keras.layers.Layer):

    def __init__(self, 
                filters,
                stage=0,
                block=0,
                kernel_size=3,
                numerical_name=False,
                stride=None,
                freeze_bn=False,
                **kwargs):
        super(Basic1D, self).__init__(**kwargs)
        
        self.filters = filters
        self.block = block
        self.kernel_size = kernel_size
        self.freeze_bn = freeze_bn
        self.stride = stride

        if stride is None:
            if block != 0 or stage == 0:
                self.stride = 1
            else:
                self.stride = 2

        if keras.backend.image_data_format() == "channels_last":
            self.axis = -1
        else:
            self.axis = 1

        if block > 0 and numerical_name:
            self.block_char = "b{}".format(block)
        else:
            self.block_char = chr(ord('a') + block)

        self.stage_char = str(stage + 2)

        
        self.zeropadding1da = keras.layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2a".format(self.stage_char, self.block_char)
        )
        self.conv1da = keras.layers.Conv1D(
            self.filters,
            self.kernel_size,
            strides=self.stride,
            use_bias=False,
            name="res{}{}_branch2a".format(self.stage_char, self.block_char),
            **parameters
        )
        self.batchnormalizationa = keras_resnet.layers.ResNetBatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2a".format(self.stage_char, self.block_char)
        )
        # self.activationa = keras.layers.Activation(
        #     "relu",
        #     name="res{}{}_branch2a_relu".format(self.stage_char, self.block_char)
        # )
        # self.zeropadding1db = keras.layers.ZeroPadding1D(
        #     padding=1,
        #     name="padding{}{}_branch2b".format(self.stage_char, self.block_char)
        # )
        # self.conv1db = keras.layers.Conv1D(
        #     self.filters,
        #     self.kernel_size,
        #     use_bias=False,
        #     name="res{}{}_branch2b".format(self.stage_char, self.block_char),
        #     **parameters
        # )
        # self.batchnormalizationb = keras_resnet.layers.ResNetBatchNormalization(
        #     axis=self.axis,
        #     epsilon=1e-5,
        #     freeze=self.freeze_bn,
        #     name="bn{}{}_branch2b".format(self.stage_char, self.block_char)
        # )
        # self.conv1dc = keras.layers.Conv1D(
        #         self.filters,
        #         1,
        #         strides=self.stride,
        #         use_bias=False,
        #         name="res{}{}_branch1".format(self.stage_char, self.block_char),
        #         **parameters
        #     )
        # self.batchnormalizationc = keras_resnet.layers.ResNetBatchNormalization(
        #         axis=self.axis,
        #         epsilon=1e-5,
        #         freeze=self.freeze_bn,
        #         name="bn{}{}_branch1".format(self.stage_char, self.block_char)
        #     )
        # self.add = keras.layers.Add(
        #     name="res{}{}".format(self.stage_char, self.block_char)
        # )
        # self.activationb = keras.layers.Activation(
        #     "relu",
        #     name="res{}{}_relu".format(self.stage_char, self.block_char)
        # )

    def call(self, inputs):
        y = self.zeropadding1da(inputs)
        y = self.conv1da(y)
        y = self.batchnormalizationa(y)
        # y = self.activationa(y)
        # y = self.zeropadding1db(y)
        # y = self.conv1db(y)
        # y = self.batchnormalizationb(y)

        # if self.block == 0:
        #     shortcut = self.conv1dc(inputs)
        #     shortcut = self.batchnormalizationc(shortcut)
        # else:
        #     shortcut = inputs

        # y = self.add([y, shortcut])
        # y = self.activationb(y)

        return y

    def get_config(self):
        return super().get_config()


# def basic_1d(
#     filters,
#     stage=0,
#     block=0,
#     kernel_size=3,
#     numerical_name=False,
#     stride=None,
#     freeze_bn=False
# ):
#     """
#     A one-dimensional basic block.

#     :param filters: the output’s feature space

#     :param stage: int representing the stage of this block (starting from 0)

#     :param block: int representing this block (starting from 0)

#     :param kernel_size: size of the kernel

#     :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

#     :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

#     :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

#     Usage:

#         >>> import keras_resnet.blocks

#         >>> keras_resnet.blocks.basic_1d(64)
#     """
#     if stride is None:
#         if block != 0 or stage == 0:
#             stride = 1
#         else:
#             stride = 2

#     if keras.backend.image_data_format() == "channels_last":
#         axis = -1
#     else:
#         axis = 1

#     if block > 0 and numerical_name:
#         block_char = "b{}".format(block)
#     else:
#         block_char = chr(ord('a') + block)

#     stage_char = str(stage + 2)

#     def f(x):
#         y = keras.layers.ZeroPadding1D(
#             padding=1, 
#             name="padding{}{}_branch2a".format(stage_char, block_char)
#         )(x)
        
#         y = keras.layers.Conv1D(
#             filters,
#             kernel_size,
#             strides=stride,
#             use_bias=False,
#             name="res{}{}_branch2a".format(stage_char, block_char),
#             **parameters
#         )(y)
        
#         y = keras_resnet.layers.ResNetBatchNormalization(
#             axis=axis,
#             epsilon=1e-5,
#             freeze=freeze_bn,
#             name="bn{}{}_branch2a".format(stage_char, block_char)
#         )(y)
        
#         y = keras.layers.Activation(
#             "relu",
#             name="res{}{}_branch2a_relu".format(stage_char, block_char)
#         )(y)

#         y = keras.layers.ZeroPadding1D(
#             padding=1,
#             name="padding{}{}_branch2b".format(stage_char, block_char)
#         )(y)
        
#         y = keras.layers.Conv1D(
#             filters,
#             kernel_size,
#             use_bias=False,
#             name="res{}{}_branch2b".format(stage_char, block_char),
#             **parameters
#         )(y)
        
#         y = keras_resnet.layers.ResNetBatchNormalization(
#             axis=axis,
#             epsilon=1e-5,
#             freeze=freeze_bn,
#             name="bn{}{}_branch2b".format(stage_char, block_char)
#         )(y)

#         if block == 0:
#             shortcut = keras.layers.Conv1D(
#                 filters,
#                 1,
#                 strides=stride,
#                 use_bias=False,
#                 name="res{}{}_branch1".format(stage_char, block_char),
#                 **parameters
#             )(x)

#             shortcut = keras_resnet.layers.ResNetBatchNormalization(
#                 axis=axis,
#                 epsilon=1e-5,
#                 freeze=freeze_bn,
#                 name="bn{}{}_branch1".format(stage_char, block_char)
#             )(shortcut)
#         else:
#             shortcut = x

#         y = keras.layers.Add(
#             name="res{}{}".format(stage_char, block_char)
#         )([y, shortcut])
        
#         y = keras.layers.Activation(
#             "relu",
#             name="res{}{}_relu".format(stage_char, block_char)
#         )(y)

#         return y

#     return f


def bottleneck_1d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    freeze_bn=False
):
    """
    A one-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.bottleneck_1d(64)
    """
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2

    if keras.backend.image_data_format() == "channels_last":
        axis = -1
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv1D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char),
            **parameters
        )(x)

        y = keras_resnet.layers.ResNetBatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = keras.layers.Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = keras.layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.Conv1D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char),
            **parameters
        )(y)

        y = keras_resnet.layers.ResNetBatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = keras.layers.Activation(
            "relu",
            name="res{}{}_branch2b_relu".format(stage_char, block_char)
        )(y)

        y = keras.layers.Conv1D(
            filters * 4,
            1,
            use_bias=False,
            name="res{}{}_branch2c".format(stage_char, block_char),
            **parameters
        )(y)

        y = keras_resnet.layers.ResNetBatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2c".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = keras.layers.Conv1D(
                filters * 4,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters
            )(x)

            shortcut = keras_resnet.layers.ResNetBatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = keras.layers.Activation(
            "relu",
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f
