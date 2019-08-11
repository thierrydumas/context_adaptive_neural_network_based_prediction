"""A library defining functions for visualizing the inputs to a Prediction Neural Network (PNN) and the parameters of the PNN."""

import numpy

import sets.arranging
import tools.tools as tls

# The functions are sorted in alphabetic order.

def visualize_context_portions(portion_above_float32, portion_left_float32, mean_training, path):
    """Arranges two masked context portions in an image and saves the image.
    
    Parameters
    ----------
    portion_above_float32 : numpy.ndarray
        3D array with data-type `numpy.float32`.
        Masked context portion located above a target
        patch. `portion_above_float32.shape[2]` is equal
        to 1.
    portion_left_float32 : numpy.ndarray
        3D array with data-type `numpy.float32`.
        Masked context portion located on the left
        side of the target patch. `portion_left_float32.shape[2]`
        is equal to 1.
    mean_training : float
        Mean pixels intensity computed over the same channel
        of different YCbCr images.
    path : str
        Path to the saved image. The path ends
        with ".png".
    
    """
    portion_above_uint8 = tls.cast_float_to_uint8(portion_above_float32 + mean_training)
    portion_left_uint8 = tls.cast_float_to_uint8(portion_left_float32 + mean_training)
    
    # `sets.arranging.arrange_context_portions` checks the shape of
    # `portion_above_uint8` and the shape of `portion_left_uint8`.
    image_uint8 = sets.arranging.arrange_context_portions(portion_above_uint8,
                                                          portion_left_uint8)
    tls.save_image(path,
                   numpy.squeeze(image_uint8, axis=2))

def visualize_flattened_context(flattened_context_float32, width_target, mean_training, path, coefficient_enlargement=None):
    """Arranges a flattened masked context in an image and saves the image.
    
    Parameters
    ----------
    flattened_context_float32 : numpy.ndarray
        1D array width data-type `numpy.float32`.
        Flattened masked context.
    width_target : int
        Width of the target patch.
    mean_training : float
        Mean pixels intensity computed over the same
        channel of different YCbCr images.
    path : str
        Path to the saved image. The path ends
        with ".png".
    coefficient_enlargement : int, optional
        Coefficient for enlarging the image along
        its first two dimensions before saving it.
        The default value is None, meaning that
        there is no enlargement of the image.
    
    """
    flattened_context_uint8 = tls.cast_float_to_uint8(flattened_context_float32 + mean_training)
    
    # If `flattened_context_uint8.ndim` is not equal to 1,
    # `sets.arranging.arrange_flattened_context` raises a
    # `ValueError` exception.
    image_uint8 = sets.arranging.arrange_flattened_context(flattened_context_uint8,
                                                           width_target)
    tls.save_image(path,
                   numpy.squeeze(image_uint8, axis=2),
                   coefficient_enlargement=coefficient_enlargement)

def visualize_weights_convolutional(weights_float32, nb_vertically, path):
    """Arranges the weight filters in a single image and saves the image.
    
    Parameters
    ----------
    weights_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Weight filters to be visualized. `weights_float32[:, :, :, i]`
        is the weight filter of index i to be visualized.
        `weights_float32.shape[2]` is equal to 1.
    nb_vertically : int
        Number of weight filters per column
        in the single image.
    path : str
        Path to the saved image. The path ends
        with ".png".
    
    """
    minimum = numpy.amin(weights_float32)
    maximum = numpy.amax(weights_float32)
    weights_uint8 = numpy.round(255.*(weights_float32 - minimum)/(maximum - minimum)).astype(numpy.uint8)
    
    # If `weights_float32.ndim` is strictly smaller
    # than 4, the swapping below raises a `ValueError`
    # exception. If `weights_float32.ndim` is strictly
    # larger than 4, `tls.visualize_channels` raises
    # a `ValueError` exception.
    channels_uint8 = numpy.swapaxes(numpy.swapaxes(numpy.swapaxes(weights_uint8, 2, 3), 1, 2),
                                    0,
                                    1)
    tls.visualize_channels(channels_uint8,
                           nb_vertically,
                           path)

def visualize_weights_fully_connected_last(weights_float32, nb_vertically, path):
    """Arranges the weight filters of the last fully-connected layer in a single image and saves the image.
    
    Parameters
    ----------
    weights_float32 : numpy.ndarray
        2D array with data-type `numpy.float32`.
        Weight filters of the last fully-connected layer.
        `weights_float32[i, :]` is the weight filter of
        index i to be visualized.
    nb_vertically : int
        Number of weight filters per column
        in the single image.
    path : str
        Path to the saved image. The path ends
        with ".png".
    
    Raises
    ------
    ValueError
        If `numpy.sqrt(weights_float32.shape[1])` is
        not a whole number.
    
    """
    # If `weights_float32.ndim` is not equal to 2,
    # the unpacking below raises a `ValueError` exception.
    (nb_filters, width_squared) = weights_float32.shape
    width_float = numpy.sqrt(width_squared).item()
    if not width_float.is_integer():
        raise ValueError('`numpy.sqrt(weights_float32.shape[1])` is not a whole number.')
    width_target = int(width_float)
    minimum = numpy.amin(weights_float32)
    maximum = numpy.amax(weights_float32)
    channels_uint8 = numpy.reshape(numpy.round(255.*(weights_float32 - minimum)/(maximum - minimum)).astype(numpy.uint8),
                                   (nb_filters, width_target, width_target, 1))
    tls.visualize_channels(channels_uint8,
                           nb_vertically,
                           path)

def visualize_weights_fully_connected_1st(weights_float32, nb_vertically, path):
    """Arranges the weight filters of the 1st fully-connected layer in a single image and saves the image.
    
    Parameters
    ----------
    weights_float32 : numpy.ndarray
        2D array with data-type `numpy.float32`.
        Weight filters of the 1st fully-connected layer.
        `weights_float32[:, i]` is the weight filter of
        index i to be visualized.
    nb_vertically : int
        Number of weight filters per column
        in the single image.
    path : str
        Path to the saved image. The path ends
        with ".png".
    
    Raises
    ------
    ValueError
        If `numpy.sqrt(weights_float32.shape[0]/5.)` is
        not a whole number.
    
    """
    # If `weights_float32.ndim` is not equal to 2,
    # the unpacking below raises a `ValueError` exception.
    (size_filter, nb_filters) = weights_float32.shape
    width_float = numpy.sqrt(size_filter/5.).item()
    if not width_float.is_integer():
        raise ValueError('`numpy.sqrt(weights_float32.shape[0]/5.)` is not a whole number.')
    width_target = int(width_float)
    minimum = numpy.amin(weights_float32)
    maximum = numpy.amax(weights_float32)
    weights_uint8 = numpy.round(255.*(weights_float32 - minimum)/(maximum - minimum)).astype(numpy.uint8)
    
    # In `sets.arranging.arrange_flattened_context`, the
    # area that does not belong to the masked context is
    # colored in white for display.
    channels_uint8 = 255*numpy.ones((nb_filters, 3*width_target, 3*width_target, 1),
                                    dtype=numpy.uint8)
    for i in range(nb_filters):
        channels_uint8[i, :, :, :] = sets.arranging.arrange_flattened_context(weights_uint8[:, i],
                                                                              width_target)
    tls.visualize_channels(channels_uint8,
                           nb_vertically,
                           path)


