import keras.backend

import keras_resnet.model


class TestResNet:
    def test_constructor(self):
        assert True


class TestResNet18:
    def test_constructor(self, x, classes):
        model = keras_resnet.model.ResNet18(x, classes)

        weights = model.trainable_weights

        parameters = [keras.backend.count_params(weight) for weight in weights]

        assert sum(parameters) == 11184650


class TestResNet34:
    def test_constructor(self, x, classes):
        model = keras_resnet.model.ResNet34(x, classes)

        weights = model.trainable_weights

        parameters = [keras.backend.count_params(weight) for weight in weights]

        assert sum(parameters) == 21296522


class TestResNet50:
    def test_constructor(self, x, classes):
        model = keras_resnet.model.ResNet50(x, classes)

        weights = model.trainable_weights

        parameters = [keras.backend.count_params(weight) for weight in weights]

        assert sum(parameters) == 58150410


class TestResNet101:
    def test_constructor(self, x, classes):
        model = keras_resnet.model.ResNet101(x, classes)

        weights = model.trainable_weights

        parameters = [keras.backend.count_params(weight) for weight in weights]

        assert sum(parameters) == 112820234


class TestResNet152:
    def test_constructor(self, x, classes):
        model = keras_resnet.model.ResNet152(x, classes)

        weights = model.trainable_weights

        parameters = [keras.backend.count_params(weight) for weight in weights]

        assert sum(parameters) == 157847050
