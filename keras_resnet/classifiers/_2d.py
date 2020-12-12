# -*- coding: utf-8 -*-

"""
keras_resnet.classifiers
~~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular residual two-dimensional classifiers.
"""

import tensorflow.tensorflow.keras.backend
import tensorflow.tensorflow.keras.layers
import tensorflow.tensorflow.keras.models
import tensorflow.tensorflow.keras.regularizers

import keras_resnet.models


class ResNet18(tensorflow.keras.models.Model):
    """
    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet18(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet18(inputs)

        outputs = tensorflow.keras.layers.Flatten()(outputs.output)

        outputs = tensorflow.keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet18, self).__init__(inputs, outputs)


class ResNet34(tensorflow.keras.models.Model):
    """
    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet34(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet34(inputs)

        outputs = tensorflow.keras.layers.Flatten()(outputs.output)

        outputs = tensorflow.keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet34, self).__init__(inputs, outputs)


class ResNet50(tensorflow.keras.models.Model):
    """
    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet50(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet50(inputs)

        outputs = tensorflow.keras.layers.Flatten()(outputs.output)

        outputs = tensorflow.keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet50, self).__init__(inputs, outputs)


class ResNet101(tensorflow.keras.models.Model):
    """
    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet101(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet101(inputs)

        outputs = tensorflow.keras.layers.Flatten()(outputs.output)

        outputs = tensorflow.keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet101, self).__init__(inputs, outputs)


class ResNet152(tensorflow.keras.models.Model):
    """
    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet152(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet152(inputs)

        outputs = tensorflow.keras.layers.Flatten()(outputs.output)

        outputs = tensorflow.keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet152, self).__init__(inputs, outputs)


class ResNet200(tensorflow.keras.models.Model):
    """
    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet200(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet200(inputs)

        outputs = tensorflow.keras.layers.Flatten()(outputs.output)

        outputs = tensorflow.keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet200, self).__init__(inputs, outputs)
