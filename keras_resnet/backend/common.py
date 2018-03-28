import keras.backend


def group_normalization(x, gamma, beta, groups, data_format=None, eps=1e-5):
    if data_format is None:
        data_format = keras.backend.image_data_format()

    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: " + str(data_format))

    if data_format == "channels_first":
        n, channels, r, c = keras.backend.shape(x)
    else:
        n, r, c, channels = keras.backend.shape(x)

    # TODO: transpose when `data_format != "channels_first"`
    x = keras.backend.reshape(x, [n, groups, channels // groups, r, c])

    mean = keras.backend.mean(x, [2, 3, 4], keepdims=True)

    var = keras.backend.var(x, [2, 3, 4], keepdims=True)

    x = (x - mean) / keras.backend.sqrt(var + eps)

    x = keras.backend.reshape(x, [n, channels, r, c])

    return x * gamma + beta
