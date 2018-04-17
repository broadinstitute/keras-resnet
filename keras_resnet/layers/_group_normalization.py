from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects


class GroupNormalization(Layer):
    """
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """
    def __init__(
        self,
        axis=-1,
        beta_constraint=None,
        beta_initializer="zeros",
        beta_regularizer=None,
        center=True,
        epsilon=1e-5,
        gamma_constraint=None,
        gamma_initializer="ones",
        gamma_regularizer=None,
        groups=32,
        scale=True,
        **kwargs
    ):
        super(GroupNormalization, self).__init__(**kwargs)

        self.axis = axis

        self.beta_constraint = constraints.get(beta_constraint)

        self.beta_initializer = initializers.get(beta_initializer)

        self.beta_regularizer = regularizers.get(beta_regularizer)

        self.center = center

        self.epsilon = epsilon

        self.gamma_constraint = constraints.get(gamma_constraint)

        self.gamma_initializer = initializers.get(gamma_initializer)

        self.gamma_regularizer = regularizers.get(gamma_regularizer)

        self.groups = groups

        self.scale = scale

        self.supports_masking = True

    def build(self, input_shape):
        dimension = input_shape[self.axis]

        if dimension is None:
            raise ValueError("Axis " + str(self.axis) + " of input tensor should have a defined dimension but the layer received an input with shape " + str(input_shape) + ".")

        if dimension < self.groups:
            raise ValueError("Number of groups (" + str(self.groups) + ") cannot be more than the number of channels (" + str(dimension) + ").")

        if dimension % self.groups != 0:
            raise ValueError("Number of groups (" + str(self.groups) + ") must be a multiple of the number of channels (" + str(dimension) + ").")

        self.input_spec = InputSpec(
            axes={
                self.axis: dimension
            },
            ndim=len(input_shape)
        )

        shape = (dimension,)

        if self.scale:
            self.gamma = self.add_weight(
                constraint=self.gamma_constraint,
                initializer=self.gamma_initializer,
                name="gamma",
                regularizer=self.gamma_regularizer,
                shape=shape
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                constraint=self.beta_constraint,
                initializer=self.beta_initializer,
                name="beta",
                regularizer=self.beta_regularizer,
                shape=shape
            )
        else:
            self.beta = None

        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        # Prepare broadcasting shape.
        ndim = len(input_shape)

        reduction_axes = list(range(len(input_shape)))

        del reduction_axes[self.axis]

        broadcast_shape = [1] * len(input_shape)

        broadcast_shape[self.axis] = input_shape[self.axis]

        reshape_group_shape = list(input_shape)

        reshape_group_shape[self.axis] = input_shape[self.axis] // self.groups

        group_shape = [-1, self.groups]

        group_shape.extend(reshape_group_shape[1:])

        group_reduction_axes = list(range(len(group_shape)))

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        inputs = K.reshape(inputs, group_shape)

        mean, variance = K.moments(inputs, group_reduction_axes[2:], keep_dims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        original_shape = [-1] + list(input_shape[1:])

        inputs = K.reshape(inputs, original_shape)

        if needs_broadcasting:
            outputs = inputs

            # In this case we must explicitly broadcast all parameters.
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)

                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)

                outputs = outputs + broadcast_beta
        else:
            outputs = inputs

            if self.scale:
                outputs = outputs * self.gamma

            if self.center:
                outputs = outputs + self.beta

        return outputs

    def get_config(self):
        config = {
            "axis": self.axis,
            "beta_constraint": constraints.serialize(
                self.beta_constraint
            ),
            "beta_initializer": initializers.serialize(
                self.beta_initializer
            ),
            "beta_regularizer": regularizers.serialize(
                self.beta_regularizer
            ),
            "center": self.center,
            "epsilon": self.epsilon,
            "gamma_constraint": constraints.serialize(
                self.gamma_constraint
            ),
            "gamma_initializer": initializers.serialize(
                self.gamma_initializer
            ),
            "gamma_regularizer": regularizers.serialize(
                self.gamma_regularizer
            ),
            "groups": self.groups,
            "scale": self.scale
        }

        base_config = super(GroupNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update(
    {
        "GroupNormalization": GroupNormalization
    }
)
