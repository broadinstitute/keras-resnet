import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Input
from keras.models import Sequential, Model
import keras_resnet.layers
import keras.utils.test_utils
import keras.backend
import keras.regularizers

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


@keras.utils.test_utils.keras_test
def test_group_normalization():
    keras.utils.test_utils.layer_test(
        keras_resnet.layers.GroupNormalization,
        input_shape=(3, 4, 2),
        kwargs={
            "beta_regularizer": keras.regularizers.l2(0.01),
            "epsilon": 0.1,
            "gamma_regularizer": keras.regularizers.l2(0.01),
            "groups": 2
        }
    )

    keras.utils.test_utils.layer_test(
        keras_resnet.layers.GroupNormalization,
        input_shape=(3, 4, 2),
        kwargs={
            "axis": 1,
            "epsilon": 0.1,
            "groups": 2
        }
    )

    keras.utils.test_utils.layer_test(
        keras_resnet.layers.GroupNormalization,
        input_shape=(3, 4, 2, 4),
        kwargs={
            "beta_initializer": "ones",
            "gamma_initializer": "ones",
            "groups": 2
        }
    )

    if keras.backend.backend() != "theano":
        keras.utils.test_utils.layer_test(
            keras_resnet.layers.GroupNormalization,
            input_shape=(3, 4, 2, 4),
            kwargs={
                "axis": 1,
                "center": False,
                "groups": 2,
                "scale": False
            }
        )


@keras.utils.test_utils.keras_test
def test_group_normalization_1d():
    model = Sequential()
    norm = keras_resnet.layers.GroupNormalization(input_shape=(10,), groups=2)
    model.add(norm)
    model.compile(loss="mse", optimizer="rmsprop")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= keras.backend.eval(norm.beta)
    out /= keras.backend.eval(norm.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


@keras.utils.test_utils.keras_test
def test_group_normalization_2d():
    model = Sequential()
    norm = keras_resnet.layers.GroupNormalization(axis=1, input_shape=(10, 6), groups=2)
    model.add(norm)
    model.compile(loss="mse", optimizer="rmsprop")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 6))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm.beta), (1, 10, 1))
    out /= np.reshape(keras.backend.eval(norm.gamma), (1, 10, 1))

    assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1.1e-1)
    assert_allclose(out.std(axis=(0, 2)), 1.0, atol=1.1e-1)


@keras.utils.test_utils.keras_test
def test_groupnorm_correctness_2d_different_groups():
    norm1 = keras_resnet.layers.GroupNormalization(axis=1, input_shape=(10, 6), groups=2)
    norm2 = keras_resnet.layers.GroupNormalization(axis=1, input_shape=(10, 6), groups=1)
    norm3 = keras_resnet.layers.GroupNormalization(axis=1, input_shape=(10, 6), groups=10)

    model = Sequential()
    model.add(norm1)
    model.compile(loss="mse", optimizer="rmsprop")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 6))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm1.beta), (1, 10, 1))
    out /= np.reshape(keras.backend.eval(norm1.gamma), (1, 10, 1))

    assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1.1e-1)
    assert_allclose(out.std(axis=(0, 2)), 1.0, atol=1.1e-1)

    model = Sequential()
    model.add(norm2)
    model.compile(loss="mse", optimizer="rmsprop")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 6))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm2.beta), (1, 10, 1))
    out /= np.reshape(keras.backend.eval(norm2.gamma), (1, 10, 1))

    assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1.1e-1)
    assert_allclose(out.std(axis=(0, 2)), 1.0, atol=1.1e-1)

    model = Sequential()
    model.add(norm3)
    model.compile(loss="mse", optimizer="rmsprop")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 6))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm3.beta), (1, 10, 1))
    out /= np.reshape(keras.backend.eval(norm3.gamma), (1, 10, 1))

    assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1.1e-1)
    assert_allclose(out.std(axis=(0, 2)), 1.0, atol=1.1e-1)


@keras.utils.test_utils.keras_test
def test_groupnorm_mode_twice():
    # This is a regression test for issue #4881 with the old
    # batch normalization functions in the Theano keras.backend.
    model = Sequential()
    model.add(keras_resnet.layers.GroupNormalization(input_shape=(10, 5, 5), axis=1, groups=2))
    model.add(keras_resnet.layers.GroupNormalization(input_shape=(10, 5, 5), axis=1, groups=2))
    model.compile(loss="mse", optimizer="sgd")

    x = np.random.normal(loc=5.0, scale=10.0, size=(20, 10, 5, 5))
    model.fit(x, x, epochs=1, verbose=0)
    model.predict(x)


@keras.utils.test_utils.keras_test
def test_groupnorm_convnet():
    model = Sequential()
    norm = keras_resnet.layers.GroupNormalization(axis=1, input_shape=(3, 4, 4), groups=3)
    model.add(norm)
    model.compile(loss="mse", optimizer="sgd")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm.beta), (1, 3, 1, 1))
    out /= np.reshape(keras.backend.eval(norm.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras.utils.test_utils.keras_test
@pytest.mark.skipif((keras.backend.backend() == "theano"), reason="Bug with theano backend")
def test_groupnorm_convnet_no_center_no_scale():
    model = Sequential()
    norm = keras_resnet.layers.GroupNormalization(axis=-1, center=False, scale=False,
                                            input_shape=(3, 4, 4), groups=2)
    model.add(norm)
    model.compile(loss="mse", optimizer="sgd")

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras.utils.test_utils.keras_test
def test_group_normalization_shared():
    """Test that a GN layer can be shared
    across different data streams.
    """
    # Test single layer reuse
    bn = keras_resnet.layers.GroupNormalization(input_shape=(10,), groups=2)
    x1 = Input(shape=(10,))
    bn(x1)

    x2 = Input(shape=(10,))
    y2 = bn(x2)

    x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
    model = Model(x2, y2)
    assert len(model.updates) == 0
    model.compile("sgd", "mse")
    model.train_on_batch(x, x)

    # Test model-level reuse
    x3 = Input(shape=(10,))
    y3 = model(x3)
    new_model = Model(x3, y3)
    assert len(model.updates) == 0
    new_model.compile("sgd", "mse")
    new_model.train_on_batch(x, x)


@keras.utils.test_utils.keras_test
def test_that_trainable_disables_updates():
    val_a = np.random.random((10, 4))
    val_out = np.random.random((10, 4))

    a = Input(shape=(4,))
    layer = keras_resnet.layers.GroupNormalization(input_shape=(4,), groups=2)
    b = layer(a)
    model = Model(a, b)

    model.trainable = False
    assert len(model.updates) == 0

    model.compile("sgd", "mse")
    assert len(model.updates) == 0

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert_allclose(x1, x2, atol=1e-7)

    model.trainable = True
    model.compile("sgd", "mse")
    assert len(model.updates) == 0

    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert np.abs(np.sum(x1 - x2)) > 1e-5

    layer.trainable = False
    model.compile("sgd", "mse")
    assert len(model.updates) == 0

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert_allclose(x1, x2, atol=1e-7)
