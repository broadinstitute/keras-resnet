import click
import keras
import keras.applications.resnet50
import keras_resnet.models
import numpy
import tensorflow

configuration = tensorflow.ConfigProto()

configuration.gpu_options.allow_growth = True

configuration.gpu_options.visible_device_list = "2"

session = tensorflow.Session(config=configuration)

keras.backend.set_session(session)

benchmarks = {
    "CIFAR-10": keras.datasets.cifar10,
    "CIFAR-100": keras.datasets.cifar100,
    "MNIST": keras.datasets.mnist
}


models = {
    "ResNet-18": keras_resnet.models.ResNet18,
    "ResNet-34": keras_resnet.models.ResNet34,
    "ResNet-50": keras_resnet.models.ResNet50,
    "ResNet-101": keras_resnet.models.ResNet101,
    "ResNet-152": keras_resnet.models.ResNet152,
    "ResNet-200": keras_resnet.models.ResNet200
}


@click.command()
@click.option(
    "--benchmark",
    default="CIFAR-10",
    type=click.Choice(
        [
            "CIFAR-10",
            "CIFAR-100",
            "MNIST"
        ]
    )
)
@click.option(
    "--name",
    default="ResNet-50",
    type=click.Choice(
        [
            "ResNet-18",
            "ResNet-34",
            "ResNet-50",
            "ResNet-101",
            "ResNet-152",
            "ResNet-200"
        ]
    )
)
def __main__(benchmark, name):
    (training_x, training_y), _ = benchmarks[benchmark].load_data()

    training_x = training_x.astype(numpy.float16)

    training_x = keras.applications.resnet50.preprocess_input(training_x)

    training_y = keras.utils.np_utils.to_categorical(training_y)

    shape, classes = training_x.shape[1:], training_y.shape[-1]

    x = keras.layers.Input(shape)

    x = models[name](x)

    y = keras.layers.Flatten()(x.output)

    y = keras.layers.Dense(classes, activation="softmax")(y)

    model = keras.models.Model(x.input, y)

    optimizer = keras.optimizers.Adam()

    loss = "categorical_crossentropy"

    metrics = [
        "accuracy"
    ]

    model.compile(optimizer, loss, metrics)

    pathname = "{}.hdf5".format(name)

    model_checkpoint = keras.callbacks.ModelCheckpoint(pathname)

    pathname = "{}.csv".format(name)

    csv_logger = keras.callbacks.CSVLogger(pathname)

    def schedule(epoch):
        if epoch < 80:
            return 0.1
        elif 80 <= epoch < 120:
            return 0.01
        else:
            return 0.001

    learning_rate_scheduler = keras.callbacks.LearningRateScheduler(schedule)

    callbacks = [
        csv_logger,
        learning_rate_scheduler,
        model_checkpoint
    ]

    model.fit(
        callbacks=callbacks,
        epochs=200,
        validation_split=0.5,
        x=training_x,
        y=training_y
    )

if __name__ == "__main__":
    __main__()
