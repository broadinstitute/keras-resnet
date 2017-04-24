keras-resnet
============

.. image:: https://travis-ci.org/broadinstitute/keras-resnet.svg?branch=master
    :target: https://travis-ci.org/broadinstitute/keras-resnet



keras-resnet is **the** Keras package for deep residual networks. It's fast *and* flexible.

A tantalizing preview of keras-resnet simplicity:

.. code-block:: python

    >>> shape, classes = (32, 32, 3), 10

    >>> x = keras.layers.Input(shape)

    >>> model = keras_resnet.models.ResNet50(x, classes)

    >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    >> (x, y), (_, _) = keras.datasets.cifar10.load_data()

    >> y = keras.utils.np_utils.to_categorical(y)

    >>> model.fit(x, y)

Installation
------------

Installation couldn’t be easier:

.. code-block:: bash

    $ pip install keras-resnet

Contributing
------------

#. Check for open issues or open a fresh issue to start a discussion around a feature idea or a bug. There is a `Contributor Friendly`_ tag for issues that should be ideal for people who are not very familiar with the codebase yet.
#. Fork `the repository`_ on GitHub to start making your changes to the **master** branch (or branch off of it).
#. Write a test which shows that the bug was fixed or that the feature works as expected.
#. Send a pull request and bug the maintainer until it gets merged and published. :) Make sure to add yourself to AUTHORS_.

.. _`the repository`: http://github.com/0x00b1/keras-resnet
.. _AUTHORS: https://github.com/0x00b1/keras-resnet/blob/master/AUTHORS.rst
.. _Contributor Friendly: https://github.com/0x00b1/keras-resnet/issues?direction=desc&labels=Contributor+Friendly&page=1&sort=updated&state=open
