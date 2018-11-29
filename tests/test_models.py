import keras_resnet.models


class TestResNet18:
    def test_constructor(self, x):
        model = keras_resnet.models.ResNet2D18(x)

        assert len(model.layers) == 87


class TestResNet34:
    def test_constructor(self, x):
        model = keras_resnet.models.ResNet2D34(x)

        assert len(model.layers) == 159


class TestResNet50:
    def test_constructor(self, x):
        model = keras_resnet.models.ResNet2D50(x)

        assert len(model.layers) == 191


class TestResNet101:
    def test_constructor(self, x):
        model = keras_resnet.models.ResNet2D101(x)

        assert len(model.layers) == 378


class TestResNet152:
    def test_constructor(self, x):
        model = keras_resnet.models.ResNet2D152(x)

        assert len(model.layers) == 565


class TestResNet200:
    def test_constructor(self, x):
        model = keras_resnet.models.ResNet2D200(x)

        assert len(model.layers) == 741
