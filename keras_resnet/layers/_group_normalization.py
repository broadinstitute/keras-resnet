import keras.backend
import keras.constraints
import keras.engine
import keras.initializers
import keras.regularizers
import keras.utils.generic_utils


class GroupNormalization(keras.engine.Layer):
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

        self.beta = None

        self.beta_constraint = keras.initializers.get(beta_constraint)

        self.beta_initializer = keras.initializers.get(beta_initializer)

        self.beta_regularizer = keras.initializers.get(beta_regularizer)

        self.center = center

        self.epsilon = epsilon

        self.gamma = None

        self.gamma_constraint = keras.initializers.get(gamma_constraint)

        self.gamma_initializer = keras.initializers.get(gamma_initializer)

        self.gamma_regularizer = keras.initializers.get(gamma_regularizer)

        self.groups = groups

        self.scale = scale

        self.supports_masking = True

    def build(self, input_shape):
        dimension = input_shape[self.axis]

        if dimension is None:
            error = "Axis " + str(self.axis) + " of input tensor should have a defined dimension but the layer received an input with shape " + str(input_shape) + "."

            raise ValueError(error)

        if dimension < self.groups:
            error = "Number of groups (" + str(self.groups) + ") cannot be more than the number of channels (" + str(dimension) + ")."

            raise ValueError(error)

        if dimension % self.groups != 0:
            error = "Number of groups (" + str(self.groups) + ") must be a multiple of the number of channels (" + str(dimension) + ")."

            raise ValueError(error)

        self.input_spec = keras.engine.InputSpec(
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

        if self.center:
            self.beta = self.add_weight(
                constraint=self.beta_constraint,
                initializer=self.beta_initializer,
                name="beta",
                regularizer=self.beta_regularizer,
                shape=shape
            )

        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = keras.backend.int_shape(inputs)

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

        inputs = keras.backend.reshape(inputs, group_shape)

        mean = keras.backend.mean(
            axis=group_reduction_axes[2:],
            keepdims=True,
            x=inputs
        )

        variance = keras.backend.var(
            axis=group_reduction_axes[2:],
            keepdims=True,
            x=inputs
        )

        inputs = (inputs - mean) / (keras.backend.sqrt(variance + self.epsilon))

        original_shape = [-1] + list(input_shape[1:])

        inputs = keras.backend.reshape(inputs, original_shape)

        if needs_broadcasting:
            outputs = inputs

            # In this case we must explicitly broadcast all parameters.
            if self.scale:
                broadcast_gamma = keras.backend.reshape(self.gamma, broadcast_shape)

                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = keras.backend.reshape(self.beta, broadcast_shape)

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
            "beta_constraint": keras.initializers.serialize(
                self.beta_constraint
            ),
            "beta_initializer": keras.initializers.serialize(
                self.beta_initializer
            ),
            "beta_regularizer": keras.initializers.serialize(
                self.beta_regularizer
            ),
            "center": self.center,
            "epsilon": self.epsilon,
            "gamma_constraint": keras.initializers.serialize(
                self.gamma_constraint
            ),
            "gamma_initializer": keras.initializers.serialize(
                self.gamma_initializer
            ),
            "gamma_regularizer": keras.initializers.serialize(
                self.gamma_regularizer
            ),
            "groups": self.groups,
            "scale": self.scale
        }

        base_config = super(GroupNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


keras.utils.generic_utils.get_custom_objects().update(
    {
        "GroupNormalization": GroupNormalization
    }
)
