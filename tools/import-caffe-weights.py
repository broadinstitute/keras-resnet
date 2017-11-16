#!/usr/bin/env python

import keras_resnet.models
import keras

import h5py
import argparse
import numpy as np


def convert_conv_weights(weights):
    return np.array(weights).transpose((2, 3, 1, 0))


def convert_dense_weights(weights, biases):
    return [np.array(weights).T, np.array(biases)]


def create_model(resnet):
    valid = ["resnet50", "resnet101", "resnet152"]
    if resnet not in valid:
        raise ValueError("Invalid resnet argument (valid: {}) : '{}'".format(valid, resnet))

    image = keras.layers.Input((None, None, 3))
    if resnet == "resnet50":
        return keras_resnet.models.ResNet50(image)
    elif resnet == "resnet101":
        return keras_resnet.models.ResNet101(image)
    elif resnet == "resnet152":
        return keras_resnet.models.ResNet152(image)


def parse_args():
    parser = argparse.ArgumentParser(description="Import caffe weights from h5 format.")
    parser.add_argument("weights", help="Path to weights (.h5) file.")
    parser.add_argument("output", help="Path to output Keras model to.")
    parser.add_argument("resnet", help="ResNet type (one of 'resnet50', 'resnet101', 'resnet152').")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # first create the model
    model = create_model(args.resnet)

    # load the caffe weights
    weights = h5py.File(args.weights).get("data")

    # port each layer
    for index, l in enumerate(model.layers):
        if isinstance(l, keras.layers.Conv2D):
            l.set_weights([convert_conv_weights(weights.get(l.name).get("0"))])
        elif isinstance(l, keras.layers.Dense):
            l.set_weights(convert_dense_weights(weights.get(l.name).get("0"), weights.get(l.name).get("1")))
        elif isinstance(l, keras.layers.BatchNormalization):
            scale_name = l.name.replace("bn", "scale")
            bn_weights = weights.get(l.name)
            scale_weights = weights.get(scale_name)

            l.set_weights([
                np.array(scale_weights.get("0")),  # gamma
                np.array(scale_weights.get("1")),  # beta
                np.array(bn_weights.get("0")),     # mean
                np.array(bn_weights.get("1")),     # variance
            ])

        print("imported layer: {}/{}".format(index, len(model.layers)), end="\r")

    print("saving...")
    model.save(args.output)
    print("done.")
