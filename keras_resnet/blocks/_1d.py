# -*- coding: utf-8 -*-

"""
keras_resnet.blocks._1d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements a number of popular one-dimensional residual blocks.
"""
<<<<<<< HEAD
import tensorflow.keras.layers
import tensorflow.keras.regularizers

=======

import keras.layers
import keras.regularizers
>>>>>>> original_regularization
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

<<<<<<< HEAD
    if tensorflow.keras.backend.image_data_format() == "channels_last":
        axis = -1
    else:
        axis = 1
=======
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
>>>>>>> original_regularization

        if block > 0 and numerical_name:
            self.block_char = "b{}".format(block)
        else:
            self.block_char = chr(ord('a') + block)

        self.stage_char = str(stage + 2)

<<<<<<< HEAD
    def f(x):
        y = tensorflow.keras.layers.ZeroPadding1D(
            padding=1, 
            name="padding{}{}_branch2a".format(stage_char, block_char)
        )(x)
        
        y = tensorflow.keras.layers.Conv1D(
            filters,
            kernel_size,
            strides=stride,
=======
        
        self.zeropadding1da = keras.layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2a".format(self.stage_char, self.block_char)
        )
        self.conv1da = keras.layers.Conv1D(
            self.filters,
            self.kernel_size,
            strides=self.stride,
>>>>>>> original_regularization
            use_bias=False,
            name="res{}{}_branch2a".format(self.stage_char, self.block_char),
            **parameters
        )
        self.batchnormalizationa = keras_resnet.layers.ResNetBatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
<<<<<<< HEAD
            freeze=freeze_bn,
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)
        
        y = tensorflow.keras.layers.Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = tensorflow.keras.layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)
        
        y = tensorflow.keras.layers.Conv1D(
            filters,
            kernel_size,
=======
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
>>>>>>> original_regularization
            use_bias=False,
            name="res{}{}_branch2b".format(self.stage_char, self.block_char),
            **parameters
        )
        self.batchnormalizationb = keras_resnet.layers.ResNetBatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
<<<<<<< HEAD
            freeze=freeze_bn,
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = tensorflow.keras.layers.Conv1D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters
            )(x)

            shortcut = keras_resnet.layers.BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = tensorflow.keras.layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])
        
        y = tensorflow.keras.layers.Activation(
=======
            freeze=self.freeze_bn,
            name="bn{}{}_branch2b".format(self.stage_char, self.block_char)
        )
        if self.block == 0 and self.stage > 0: #Dotted line connections in ResNet paper
            self.conv1dc = keras.layers.Conv1D(
                    self.filters,
                    1,
                    strides=self.stride,
                    use_bias=False,
                    name="res{}{}_branch1".format(self.stage_char, self.block_char),
                    **parameters
                )
            self.batchnormalizationc = keras_resnet.layers.ResNetBatchNormalization(
                    axis=self.axis,
                    epsilon=1e-5,
                    freeze=self.freeze_bn,
                    name="bn{}{}_branch1".format(self.stage_char, self.block_char)
                )
        self.add = keras.layers.Add(
            name="res{}{}".format(self.stage_char, self.block_char)
        )
        self.activationb = keras.layers.Activation(
>>>>>>> original_regularization
            "relu",
            name="res{}{}_relu".format(self.stage_char, self.block_char)
        )

    def call(self, inputs):
        y = self.zeropadding1da(inputs)
        y = self.conv1da(y)
        y = self.batchnormalizationa(y)
        y = self.activationa(y)
        y = self.zeropadding1db(y)
        y = self.conv1db(y)
        y = self.batchnormalizationb(y)

        if self.block == 0 and self.stage > 0: #Dotted line connections in ResNet paper
            shortcut = self.conv1dc(inputs)
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
<<<<<<< HEAD
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2

    if tensorflow.keras.backend.image_data_format() == "channels_last":
        axis = -1
    else:
        axis = 1
=======
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
>>>>>>> original_regularization

        if block > 0 and numerical_name:
            self.block_char = "b{}".format(block)
        else:
            self.block_char = chr(ord('a') + block)

        self.stage_char = str(stage + 2)

<<<<<<< HEAD
    def f(x):
        y = tensorflow.keras.layers.Conv1D(
            filters,
=======
    
        self.conv1da = keras.layers.Conv1D(
            self.filters,
>>>>>>> original_regularization
            1,
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

<<<<<<< HEAD
        y = tensorflow.keras.layers.Activation(
=======
        self.activationa = keras.layers.Activation(
>>>>>>> original_regularization
            "relu",
            name="res{}{}_branch2a_relu".format(self.stage_char, self.block_char)
        )

<<<<<<< HEAD
        y = tensorflow.keras.layers.ZeroPadding1D(
=======
        self.zeropadding1da = keras.layers.ZeroPadding1D(
>>>>>>> original_regularization
            padding=1,
            name="padding{}{}_branch2b".format(self.stage_char, self.block_char)
        )

<<<<<<< HEAD
        y = tensorflow.keras.layers.Conv1D(
            filters,
            kernel_size,
=======
        self.conv1db = keras.layers.Conv1D(
            self.filters,
            self.kernel_size,
>>>>>>> original_regularization
            use_bias=False,
            name="res{}{}_branch2b".format(self.stage_char, self.block_char),
            **parameters
        )

        self.batchnormalizationb = keras_resnet.layers.ResNetBatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2b".format(self.stage_char, self.block_char)
        )

<<<<<<< HEAD
        y = tensorflow.keras.layers.Activation(
=======
        self.activationb = keras.layers.Activation(
>>>>>>> original_regularization
            "relu",
            name="res{}{}_branch2b_relu".format(self.stage_char, self.block_char)
        )

<<<<<<< HEAD
        y = tensorflow.keras.layers.Conv1D(
            filters * 4,
=======
        self.conv1dc = keras.layers.Conv1D(
            self.filters * 4,
>>>>>>> original_regularization
            1,
            use_bias=False,
            name="res{}{}_branch2c".format(self.stage_char, self.block_char),
            **parameters
        )

        self.batchnormalizationc = keras_resnet.layers.ResNetBatchNormalization(
            axis=self.axis,
            epsilon=1e-5,
            freeze=self.freeze_bn,
            name="bn{}{}_branch2c".format(self.stage_char, self.block_char)
        )

<<<<<<< HEAD
        if block == 0:
            shortcut = tensorflow.keras.layers.Conv1D(
                filters * 4,
=======
        if self.block == 0 and self.stage > 0: #Dotted line connections in ResNet paper
            self.conv1dd = keras.layers.Conv1D(
                self.filters * 4,
>>>>>>> original_regularization
                1,
                strides=self.stride,
                use_bias=False,
                name="res{}{}_branch1".format(self.stage_char, self.block_char),
                **parameters
            )

            self.batchnormalizationd = keras_resnet.layers.ResNetBatchNormalization(
                axis=self.axis,
                epsilon=1e-5,
                freeze=self.freeze_bn,
                name="bn{}{}_branch1".format(self.stage_char, self.block_char)
            )

<<<<<<< HEAD
        y = tensorflow.keras.layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = tensorflow.keras.layers.Activation(
=======
        self.add = keras.layers.Add(
            name="res{}{}".format(self.stage_char, self.block_char)
        )

        self.activationc = keras.layers.Activation(
>>>>>>> original_regularization
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

        if self.block == 0 and self.stage > 0: #Dotted line connections in ResNet paper
            shortcut = self.conv1dd(inputs)
            shortcut = self.batchnormalizationd(shortcut)
        else: #Solid line connections in ResNet paper
            shortcut = inputs

        y = self.add([y, shortcut])
        y = self.activationc(y)

        return y