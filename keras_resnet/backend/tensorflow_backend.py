import tensorflow


def resize(image, output_shape):
    """
    Resize an image or images to match a certain size.

    :param image: Input image or images with the shape:

        (rows, columns, channels)

    or:

        (batch, rows, columns, channels).

    :param output_shape: Shape of the output image:

        (rows, columns).

    :return: If an image is provided a resized image with the shape:

        (resized rows, resized columns, channels)

    is returned.

    If more than one image is provided then a batch of resized images with
    the shape:

        (batch size, resized rows, resized columns, channels)

    are returned.
    """
    return tensorflow.image.resize_images(image, output_shape)
