# -*- coding: utf-8 -*-

"""
keras_resnet.models
~~~~~~~~~~~~~~~~~~~

This module implements popular residual models.
"""

from ._1d import (
    ResNet1D,
    ResNet1D18,
    ResNet1D34,
    ResNet1D50,
    ResNet1D101,
    ResNet1D152,
    ResNet1D200
)

from ._2d import (
    ResNet2D,
    ResNet2D18,
    ResNet2D34,
    ResNet2D50,
    ResNet2D101,
    ResNet2D152,
    ResNet2D200
)

from ._3d import (
    ResNet3D,
    ResNet3D18,
    ResNet3D34,
    ResNet3D50,
    ResNet3D101,
    ResNet3D152,
    ResNet3D200
)

from ._time_distributed_2d import (
    TimeDistributedResNet,
    TimeDistributedResNet18,
    TimeDistributedResNet34,
    TimeDistributedResNet50,
    TimeDistributedResNet101,
    TimeDistributedResNet152,
    TimeDistributedResNet200
)
