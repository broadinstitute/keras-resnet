import keras.layers
import pytest


@pytest.fixture(scope="module")
def x():
    shape = (32, 32, 3)

    return keras.layers.Input(shape)
