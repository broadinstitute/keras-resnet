# -*- coding: utf-8 -*-

"""
keras_resnet.models
~~~~~~~~~~~~~~~~~~~

This module implements popular residual models.
"""

from ._2d import (
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet200,
    resnet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet200
)

from ._time_distributed_2d import (
    TimeDistributedResNet,
    TimeDistributedResNet18,
    TimeDistributedResNet34,
    TimeDistributedResNet50,
    TimeDistributedResNet101,
    TimeDistributedResNet152,
    TimeDistributedResNet200,
    time_distributed_resnet,
    time_distributed_resnet18,
    time_distributed_resnet34,
    time_distributed_resnet50,
    time_distributed_resnet101,
    time_distributed_resnet152,
    time_distributed_resnet200
)
