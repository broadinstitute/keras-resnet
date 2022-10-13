# -*- coding: utf-8 -*-

"""
keras_resnet.models._1d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular one-dimensional residual models.
"""

import tensorflow.keras.backend
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.regularizers

import keras_resnet.blocks
import keras_resnet.layers


class ResNet1D(tensorflow.keras.Model):
    """
    Constructs a `tensorflow.keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_1d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_1d

        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(
        self,
        blocks,
        block,
        include_top=True,
        classes=1000,
        freeze_bn=True,
        numerical_names=None,
        *args,
        **kwargs
    ):
        super(ResNet1D, self).__init__(*args, **kwargs)
        self.classes = classes
        self.include_top = include_top

        if tensorflow.keras.backend.image_data_format() == "channels_last":
            axis = -1
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        self.zeropad1 = tensorflow.keras.layers.ZeroPadding1D(padding=3, name="padding_conv1")
        self.conv1 = tensorflow.keras.layers.Conv1D(64, 7, strides=2, use_bias=False, name="conv1")
        self.rnbn1 = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")
        self.relu1 = tensorflow.keras.layers.Activation("relu", name="conv1_relu")
        self.maxpool1 = tensorflow.keras.layers.MaxPooling1D(3, strides=2, padding="same", name="pool1")

        features = 64
        self.lyrs = []
        self.iters = []

        for stage_id, iterations in enumerate(blocks):
            self.iters.append(iterations)
            for block_id in range(iterations):
                lyr = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )
                self.lyrs.append (lyr)
                self.layers.append (lyr)
            features *= 2

        self.glopoollast = tensorflow.keras.layers.GlobalAveragePooling1D(name="pool5")
        self.fclast = tensorflow.keras.layers.Dense(classes, activation="softmax", name="fc1000")

    
    def call(self, inputs):
        x = self.zeropad1(inputs)
        x = self.conv1(x)
        x = self.rnbn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        outputs = []
        i = 0
        while len(self.lyrs) > 0:
            x = self.lyrs[0](x)
            self.lyrs.pop()
            i += 1
            try:
                if i == self.iters[0]:
                    outputs.append(x)
                    self.iters.pop()
                    i = 0
            except:
                pass

        if self.include_top:
            assert self.classes > 0
            x = self.glopoollast(x)
            return self.fclast(x)
        else:
            return outputs


class ResNet1D18(ResNet1D):
    """
    Constructs a `tensorflow.keras.models.Model` according to the ResNet18 specifications.

    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 3), 1000

        >>> x = tensorflow.keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet18(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """ 
    
    def __init__(self, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]
        
        super(ResNet1D18, self).__init__(
            blocks,
            block=keras_resnet.blocks.basic_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

    def call (self, inputs):
        return super(ResNet1D18, self).call(inputs)


class ResNet1D34(ResNet1D):
    """
    Constructs a `tensorflow.keras.models.Model` according to the ResNet34 specifications.
    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = tensorflow.keras.layers.Input(shape)
        >>> model = keras_resnet.models.ResNet34(x, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(ResNet1D34, self).__init__(
            blocks,
            block=keras_resnet.blocks.basic_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

    def call (self, inputs):
        return super(ResNet1D34, self).call(inputs)


class ResNet1D50(ResNet1D):
    """
    Constructs a `tensorflow.keras.models.Model` according to the ResNet50 specifications.
    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = tensorflow.keras.layers.Input(shape)
        >>> model = keras_resnet.models.ResNet50(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet1D50, self).__init__(
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

    def call (self, inputs):
        return super(ResNet1D50, self).call(inputs)


class ResNet1D101(ResNet1D):
    """
    Constructs a `tensorflow.keras.models.Model` according to the ResNet101 specifications.
    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = tensorflow.keras.layers.Input(shape)
        >>> model = keras_resnet.models.ResNet101(x, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(ResNet1D101, self).__init__(
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

    def call (self, inputs):
            return super(ResNet1D101, self).call(inputs)


class ResNet1D152(ResNet1D):
    """
    Constructs a `tensorflow.keras.models.Model` according to the ResNet152 specifications.
    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = tensorflow.keras.layers.Input(shape)
        >>> model = keras_resnet.models.ResNet152(x, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet1D152, self).__init__(
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

    def call (self, inputs):
        return super(ResNet1D152, self).call(inputs)

class ResNet1D200(ResNet1D):
    """
    Constructs a `tensorflow.keras.models.Model` according to the ResNet200 specifications.
    :param inputs: input tensor (e.g. an instance of `tensorflow.keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = tensorflow.keras.layers.Input(shape)
        >>> model = keras_resnet.models.ResNet200(x, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet1D200, self).__init__(
            blocks,
            numerical_names=numerical_names,
            block=keras_resnet.blocks.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

    def call (self, inputs):
        return super(ResNet1D200, self).call(inputs)