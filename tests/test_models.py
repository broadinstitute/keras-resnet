import keras_resnet.models


def test_resnet18(x):
    model = keras_resnet.models.resnet18(x)

    assert len(model.layers) == 88


def test_resnet34(x):
    model = keras_resnet.models.resnet34(x)

    assert len(model.layers) == 160


def test_resnet50(x):
    model = keras_resnet.models.resnet50(x)

    assert len(model.layers) == 192


def test_resnet101(x):
    model = keras_resnet.models.resnet101(x)

    assert len(model.layers) == 379


def test_resnet152(x):
    model = keras_resnet.models.resnet152(x)

    assert len(model.layers) == 566


def test_resnet200(x):
    model = keras_resnet.models.resnet200(x)

    assert len(model.layers) == 742
