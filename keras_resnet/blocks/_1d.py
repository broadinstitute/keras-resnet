# -*- coding: utf-8 -*-

"""
keras_resnet.blocks._1d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements a number of popular one-dimensional residual blocks.
"""

import keras.layers
import keras.regularizers
import keras_resnet.layers

parameters = {"kernel_initializer": "he_normal"}

class Basic1D(keras.layers.Layer):
    """
    A one-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    """

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
        self.stage = stage
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
        self.batchnormalizationa = keras_resnet.layers.BatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2a".format(self.stage_char, self.block_char)
        )
        self.activationa = keras.layers.Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(self.stage_char, self.block_char)
        )
        self.zeropadding1db = keras.layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(self.stage_char, self.block_char)
        )
        self.conv1db = keras.layers.Conv1D(
            self.filters,
            self.kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(self.stage_char, self.block_char),
            **parameters
        )
        self.batchnormalizationb = keras_resnet.layers.BatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2b".format(self.stage_char, self.block_char)
        )
        if self.block == 0: #Dotted line connections in ResNet paper
            self.conv1dc = keras.layers.Conv1D(
                    self.filters,
                    1,
                    strides=self.stride,
                    use_bias=False,
                    name="res{}{}_branch1".format(self.stage_char, self.block_char),
                    **parameters
                )
            self.batchnormalizationc = keras_resnet.layers.BatchNormalization(
                    axis=self.axis,
                    epsilon=1e-5,
                    freeze=self.freeze_bn,
                    name="bn{}{}_branch1".format(self.stage_char, self.block_char)
                )
        self.add = keras.layers.Add(
            name="res{}{}".format(self.stage_char, self.block_char)
        )
        self.activationb = keras.layers.Activation(
            "relu",
            name="res{}{}_relu".format(self.stage_char, self.block_char)
        )

    def call(self, inputs):
        y = self.zeropadding1da(inputs) #padding2a_branch2a
        y = self.conv1da(y) #res2a_branch2a
        y = self.batchnormalizationa(y) #bn2a_branch2a
        y = self.activationa(y) #res2a_branch2a_relu
        y = self.zeropadding1db(y) #padding2a_branch2b
        y = self.conv1db(y) #res2a_branch2b
        y = self.batchnormalizationb(y) #bn2a_branch2b

        if self.block == 0: #Dotted line connections in ResNet paper
            shortcut = self.conv1dc(inputs) #!!!res2a_branch1
            shortcut = self.batchnormalizationc(shortcut)
        else: #Solid line connections in ResNet paper
            shortcut = inputs

        y = self.add([y, shortcut])
        y = self.activationb(y)

        return y


class Bottleneck1D(keras.layers.Layer):
    """
    A one-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    """
    def __init__(self, 
                filters,
                stage=0,
                block=0,
                kernel_size=3,
                numerical_name=False,
                stride=None,
                freeze_bn=False,
                **kwargs):
        super(Bottleneck1D, self).__init__(**kwargs)
        
        self.filters = filters
        self.stage = stage
        self.block = block
        self.kernel_size = kernel_size
        self.freeze_bn = freeze_bn
        self.stride = stride

        if stride is None:
            self.stride = 1 if block != 0 or stage == 0 else 2

        if keras.backend.image_data_format() == "channels_last":
            self.axis = -1
        else:
            self.axis = 1

        if block > 0 and numerical_name:
            self.block_char = "b{}".format(block)
        else:
            self.block_char = chr(ord('a') + block)

        self.stage_char = str(stage + 2)

    
        self.conv1da = keras.layers.Conv1D(
            self.filters,
            1,
            strides=self.stride,
            use_bias=False,
            name="res{}{}_branch2a".format(self.stage_char, self.block_char),
            **parameters
        )

        self.batchnormalizationa = keras_resnet.layers.BatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2a".format(self.stage_char, self.block_char)
        )

        self.activationa = keras.layers.Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(self.stage_char, self.block_char)
        )

        self.zeropadding1da = keras.layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(self.stage_char, self.block_char)
        )

        self.conv1db = keras.layers.Conv1D(
            self.filters,
            self.kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(self.stage_char, self.block_char),
            **parameters
        )

        self.batchnormalizationb = keras_resnet.layers.BatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2b".format(self.stage_char, self.block_char)
        )

        self.activationb = keras.layers.Activation(
            "relu",
            name="res{}{}_branch2b_relu".format(self.stage_char, self.block_char)
        )

        self.conv1dc = keras.layers.Conv1D(
            self.filters * 4,
            1,
            use_bias=False,
            name="res{}{}_branch2c".format(self.stage_char, self.block_char),
            **parameters
        )

        self.batchnormalizationc = keras_resnet.layers.BatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2c".format(self.stage_char, self.block_char)
        )

        self.conv1dd = keras.layers.Conv1D(
            self.filters * 4,
            1,
            strides=self.stride,
            use_bias=False,
            name="res{}{}_branch1".format(self.stage_char, self.block_char),
            **parameters
        )

        self.batchnormalizationd = keras_resnet.layers.BatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch1".format(self.stage_char, self.block_char)
        )

        self.add = keras.layers.Add(
            name="res{}{}".format(self.stage_char, self.block_char)
        )

        self.activationc = keras.layers.Activation(
            "relu",
            name="res{}{}_relu".format(self.stage_char, self.block_char)
        )

    def call(self, inputs):
        y = self.conv1da(inputs)
        y = self.batchnormalizationa(y)
        y = self.activationa(y)
        y = self.zeropadding1da(y)
        y = self.conv1db(y)
        y = self.batchnormalizationb(y)
        y = self.activationb(y)
        y = self.conv1dc(y)
        y = self.batchnormalizationc(y)

        if self.block == 0: #Dotted line connections in ResNet paper
            shortcut = self.conv1dd(inputs)
            shortcut = self.batchnormalizationd(shortcut)
        else: #Solid line connections in ResNet paper
            shortcut = inputs
        
        y = self.add([y, shortcut])
        y = self.activationc(y)

        return y