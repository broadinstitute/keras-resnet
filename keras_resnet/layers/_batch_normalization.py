import keras


class ResNetBatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(ResNetBatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
        if self.freeze:
            kwargs['training'] = False
        return super(ResNetBatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        config = super(ResNetBatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config
