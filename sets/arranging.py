"""A library containing functions for arranging the PNN context in an image."""

import numpy

# The functions are sorted in alphabetic order.

def arrange_context_portions(portion_above_uint8, portion_left_uint8):
    """Arranges the context portion located above a target patch and the context portion located on the left side of this target patch in an image.
    
    Parameters
    ----------
    portion_above_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Context portion located above a target patch.
        `portion_above_uint8.shape[2]` is equal to 1.
    portion_left_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Context portion located on the left side of
        this target patch. `portion_left_uint8.shape[2]`
        is equal to 1.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image containing the two context portions. The
        3rd array dimension is equal to 1.
    
    Raises
    ------
    TypeError
        If `portion_above_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        If `portion_left_uint8.dtype` is not equal to `numpy.uint8`.
    
    """
    if portion_above_uint8.dtype != numpy.uint8:
        raise TypeError('`portion_above_uint8.dtype` is not equal to `numpy.uint8`.')
    if portion_left_uint8.dtype != numpy.uint8:
        raise TypeError('`portion_left_uint8.dtype` is not equal to `numpy.uint8`.')
    width_target = portion_above_uint8.shape[0]
    image_uint8 = 255*numpy.ones((3*width_target, 3*width_target, 1), dtype=numpy.uint8)
    
    # If `portion_above_uint8.shape` is not equal to
    # (`width_target`, `3*width_target`, 1), the copy
    # below raises a `ValueError` exception.
    image_uint8[0:width_target, :, :] = portion_above_uint8
    
    # If `portion_left_uint8.shape` is not equal to
    # (`2*width_target`, `width_target`, 1), the copy
    # below raises a `ValueError` exception.
    image_uint8[width_target:, 0:width_target, :] = portion_left_uint8
    return image_uint8

def arrange_flattened_context(flattened_context_uint8, width_target):
    """Arranges the flattened context in an image.
    
    Parameters
    ----------
    flattened_context_uint8 : numpy.ndarray
        1D array with data-type `numpy.uint8`.
        Flattened context.
    width_target : int
        Width of the target patch.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image containing the flattened context
        after the arrangement. The 3rd array
        dimension is equal to 1.
    
    Raises
    ------
    TypeError
        If `flattened_context_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `flattened_context_uint8.ndim` is not equal to 1.
    
    """
    if flattened_context_uint8.dtype != numpy.uint8:
        raise TypeError('`flattened_context_uint8.dtype` is not equal to `numpy.uint8`.')
    if flattened_context_uint8.ndim != 1:
        raise ValueError('`flattened_context_uint8.ndim` is not equal to 1.')
    
    # If `flattened_context_uint8.size` is not equal to
    # `5*width_target**2`, the change of shape below raises
    # a `ValueError` exception.
    image_uint8 = 255*numpy.ones((3*width_target, 3*width_target, 1), dtype=numpy.uint8)
    image_uint8[0:width_target, :, :] = numpy.reshape(flattened_context_uint8[0:3*width_target**2],
                                                      (width_target, 3*width_target, 1))
    image_uint8[width_target:, 0:width_target, :] = numpy.reshape(flattened_context_uint8[3*width_target**2:],
                                                                  (2*width_target, width_target, 1))
    return image_uint8


