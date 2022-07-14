from . import layers

custom_objects = {
    'BatchNormalization': layers.ResNetBatchNormalization,
}
