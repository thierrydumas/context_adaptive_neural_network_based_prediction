"""A library defining a common pipeline for data extraction.

This pipeline extracts several target patches, each
paired with its two context portions, from the same
channel of different images and preprocesses them.

"""

import numpy

# The functions are sorted in alphabetic order.

def extract_context_portions_target_from_channel_numpy(channel_single_or_pair_uint8, width_target, row_1st, col_1st):
    """Extracts a target patch and its two context portions from the image channel.
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 1, `channel_single_or_pair_uint8` is an image channel.
        If `channel_single_or_pair_uint8.shape[2]` is equal to 2,
        `channel_single_or_pair_uint8[:, :, 0:1]` is an image
        channel and `channel_single_or_pair_uint8[:, :, 1:2]`
        is this image channel with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1st : int
        Row of the 1st context pixel in the image channel.
        The 1st context pixel is the pixel at the top-left
        of the context portion located above the target patch.
    col_1st : int
        Column of the 1st context pixel in the image channel.
    
    Returns
    -------
    tuple
        numpy.ndarray
            3D array with data-type `numpy.uint8`.
            Context portion located above the target patch.
            The array shape is equal to (`width_target`, `3*width_target`,
            1). If `channel_single_or_pair_uint8.shape[2]` is
            equal  to 1, the context portion located above the
            target patch is extracted from the image channel. If
            `channel_single_or_pair_uint8.shape[2]` is equal to
            2, it is extracted from the image channel with HEVC
            compression artifacts.
        numpy.ndarray
            3D array with data-type `numpy.uint8`.
            Context portion located on the left side of the
            target patch. The array shape is equal to (`2*width_target`,
            `width_target`, 1). If `channel_single_or_pair_uint8.shape[2]`
            is equal to 1, the context portion located on the
            left side of the target patch is extracted from the
            image channel . If `channel_single_or_pair_uint8.shape[2]`
            is equal to 2, it is extracted from the image channel with
            HEVC compression artifacts.
        numpy.ndarray
            3D array with data-type `numpy.uint8`.
            Target patch. The array shape is equal to (`width_target`,
            `width_target`, 1). The target patch is extracted from
            the image channel.
    
    Raises
    ------
    TypeError
        If `channel_single_or_pair_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `width_target` is not positive.
    ValueError
        If `row_1st` is not positive.
    ValueError
        If `col_1st` is not positive.
    ValueError
        `row_1st + 3*width_target` is not smaller than
        `channel_single_or_pair_uint8.shape[0]`.
    ValueError
        `col_1st + 3*width_target` is not smaller than
        `channel_single_or_pair_uint8.shape[1]`.
    ValueError
        If `channel_single_or_pair_uint8.shape[2]` does not belong to {1, 2}.
    
    """
    if channel_single_or_pair_uint8.dtype != numpy.uint8:
        raise TypeError('`channel_single_or_pair_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If `channel_single_or_pair_uint8.ndim` is not equal
    # to 3, the unpacking below raises a `ValueError` exception.
    (height_channel, width_channel, nb_channels) = channel_single_or_pair_uint8.shape
    
    # If the five checks below did not exist, `extract_context_portions_target_from_channel_numpy`
    # would not crash if the slicings below go out of bounds.
    if width_target < 0:
        raise ValueError('`width_target` is not positive.')
    if row_1st < 0:
        raise ValueError('`row_1st` is not positive.')
    if col_1st < 0:
        raise ValueError('`col_1st` is not positive.')
    if row_1st + 3*width_target > height_channel:
        raise ValueError('`row_1st + 3*width_target` is not smaller than `channel_single_or_pair_uint8.shape[0]`.')
    if col_1st + 3*width_target > width_channel:
        raise ValueError('`col_1st + 3*width_target` is not smaller than `channel_single_or_pair_uint8.shape[1]`.')
    if nb_channels == 1 or nb_channels == 2:
        i = nb_channels - 1
    else:
        raise ValueError('`channel_single_or_pair_uint8.shape[2]` does not belong to {1, 2}.')
    portion_above_uint8 = channel_single_or_pair_uint8[row_1st:row_1st + width_target, col_1st:col_1st + 3*width_target, i:i + 1]
    portion_left_uint8 = channel_single_or_pair_uint8[row_1st + width_target:row_1st + 3*width_target, col_1st:col_1st + width_target, i:i + 1]
    target_uint8 = channel_single_or_pair_uint8[row_1st + width_target:row_1st + 2*width_target, col_1st + width_target:col_1st + 2*width_target, 0:1]
    return (portion_above_uint8, portion_left_uint8, target_uint8)

def extract_context_portions_targets_from_channel_numpy(channel_single_or_pair_uint8, width_target, row_1sts, col_1sts):
    """Extracts several target patches, each paired with its two context portions, from the image channel.
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 1, `channel_single_or_pair_uint8` is an image channel.
        If `channel_single_or_pair_uint8.shape[2]` is equal to 2,
        `channel_single_or_pair_uint8[:, :, 0:1]` is an image channel
        and `channel_single_or_pair_uint8[:, :, 1:2]` is this image
        channel with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    
    Returns
    -------
    tuple
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Context portions, each being located above a
            different target patch. The array shape is equal
            to (`row_1sts.size`, `width_target`, `3*width_target`, 1).
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Context portions, each being located on the left
            side of a different target patch. The array shape is
            equal to (`row_1sts.size`, `2*width_target`, `width_target`, 1).
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Target patches. The array shape is equal to (`row_1sts.size`,
            `width_target`, `width_target`, 1).
    
    Raises
    ------
    TypeError
        If `row_1sts.dtype` is not smaller than `numpy.integer`
        in type hierarchy.
    TypeError
        If `col_1sts.dtype` is not smaller than `numpy.integer`
        in type hierarchy.
    ValueError
        If `col_1sts.size` is not equal to `row_1sts.size`.
    
    """
    # The check below is crucial because, if the elements
    # of `row_1sts` are floats or booleans for instance, the
    # slicings below fire warnings instead of raising exceptions.
    if not numpy.issubdtype(row_1sts.dtype, numpy.integer):
        raise TypeError('`row_1sts.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    if not numpy.issubdtype(col_1sts.dtype, numpy.integer):
        raise TypeError('`col_1sts.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    size_row_1sts = row_1sts.size
    
    # If the check below did not exist, `extract_context_portions_targets_from_channel_numpy`
    # would not crash if `col_1sts.size` is strictly larger than `row_1sts.size`.
    if col_1sts.size != size_row_1sts:
        raise ValueError('`col_1sts.size` is not equal to `row_1sts.size`.')
    portions_above_uint8 = numpy.zeros((size_row_1sts, width_target, 3*width_target, 1), dtype=numpy.uint8)
    portions_left_uint8 = numpy.zeros((size_row_1sts, 2*width_target, width_target, 1), dtype=numpy.uint8)
    targets_uint8 = numpy.zeros((size_row_1sts, width_target, width_target, 1), dtype=numpy.uint8)
    for i in range(size_row_1sts):
        
        # If `row_1sts.ndim` is not equal to 1, `row_1sts[i].item()`
        # raises a `ValueError` exception. Similarly, if `col_1sts.ndim`
        # is not equal to 1, `col_1sts[i].item()` raises a `ValueError`
        # exception.
        (portions_above_uint8[i, :, :, :], portions_left_uint8[i, :, :, :], targets_uint8[i, :, :, :]) = \
            extract_context_portions_target_from_channel_numpy(channel_single_or_pair_uint8,
                                                               width_target,
                                                               row_1sts[i].item(),
                                                               col_1sts[i].item())
    return (portions_above_uint8, portions_left_uint8, targets_uint8)

def extract_context_portions_targets_from_channels_numpy(channels_single_or_pair_uint8, width_target, row_1sts, col_1sts):
    """Extracts several target patches, each paired with its two context portions, from the same channel of each image.
    
    Parameters
    ----------
    channels_single_or_pair_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. If `channels_single_or_pair_uint8.shape[3]` is equal to 1,
        `channels_single_or_pair_uint8[i, :, :, :]` is the channel of the image of
        index i. If `channels_single_or_pair_uint8.shape[3]` is equal to 2,
        `channels_single_or_pair_uint8[i, :, :, 0:1]` is the channel of the
        image of index i and `channels_single_or_pair_uint8[i, :, :, 1:2]`
        is the channel of the image of index i with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    
    Returns
    -------
    tuple
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Context portions, each being located above a
            different target patch. The array shape is equal
            to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
            `width_target`, `3*width_target`, 1).
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Context portions, each being located on the left
            side of a different target patch. The array shape is
            equal to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
            `2*width_target`, `width_target`, 1).
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Target patches. The array shape is equal to
            (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`, `width_target`,
            `width_target`, 1).
    
    """
    nb_images = channels_single_or_pair_uint8.shape[0]
    size_row_1sts = row_1sts.size
    portions_above_uint8 = numpy.zeros((nb_images*size_row_1sts, width_target, 3*width_target, 1), dtype=numpy.uint8)
    portions_left_uint8 = numpy.zeros((nb_images*size_row_1sts, 2*width_target, width_target, 1), dtype=numpy.uint8)
    targets_uint8 = numpy.zeros((nb_images*size_row_1sts, width_target, width_target, 1), dtype=numpy.uint8)
    for i in range(nb_images):
        (
            portions_above_uint8[i*size_row_1sts:(i + 1)*size_row_1sts, :, :, :],
            portions_left_uint8[i*size_row_1sts:(i + 1)*size_row_1sts, :, :, :],
            targets_uint8[i*size_row_1sts:(i + 1)*size_row_1sts, :, :, :]
        ) = extract_context_portions_targets_from_channel_numpy(channels_single_or_pair_uint8[i, :, :, :],
                                                                width_target,
                                                                row_1sts,
                                                                col_1sts)
    return (portions_above_uint8, portions_left_uint8, targets_uint8)

def extract_context_portions_targets_from_channels_plus_preprocessing(channels_single_or_pair_uint8, width_target, row_1sts, col_1sts,
                                                                      mean_training, tuple_width_height_masks, is_fully_connected):
    """Extracts several target patches, each paired with its two context portions, from the same channel of each image and preprocesses them.
    
    Parameters
    ----------
    channels_single_or_pair_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. If `channels_single_or_pair_uint8.shape[3]`
        is equal to 1, `channels_single_or_pair_uint8[i, :, :, :]` is
        the channel of the image of index i. If `channels_single_or_pair_uint8.shape[3]`
        is equal to 2, `channels_single_or_pair_uint8[i, :, :, 0:1]`
        is the channel of the image of index i and `channels_single_or_pair_uint8[i, :, :, 1:2]`
        is the channel of the image of index i with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of the 1st
        pixel of the context portion located above the target patch
        of index i.
    mean_training : float
        Mean pixels intensity computed over the same channel
        of different YCbCr images.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the width of the mask
        that covers the right side of the context portion located
        above the target patch. The 2nd integer in this tuple is
        the height of the mask that covers the bottom of the context
        portion located on the left side of the target patch.
    is_fully_connected : bool
        Is PNN fully-connected?
    
    Returns
    -------
    tuple
        If `is_fully_connected` is True,
            numpy.ndarray
                2D array with data-type `numpy.float32`.
                Flattened masked contexts. The array shape is
                equal to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
                `5*width_target**2`).
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Target patches. The array shape is equal to
                (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`, `width_target`,
                `width_target`, 1).
        Otherwise,
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Masked context portions, each being located
                above a different target patch. The array shape 
                is equal to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
                `width_target`, `3*width_target`, 1).
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Masked context portions, each being located on
                the left side of a different target patch. The
                array shape is equal to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
                `2*width_target`, `width_target`, 1).
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Target patches. The array shape is equal to
                (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`, `width_target`,
                `width_target`, 1).
    
    """
    (portions_above_uint8, portions_left_uint8, targets_uint8) = \
        extract_context_portions_targets_from_channels_numpy(channels_single_or_pair_uint8,
                                                             width_target,
                                                             row_1sts,
                                                             col_1sts)
    return preprocess_context_portions_targets_numpy(portions_above_uint8,
                                                     portions_left_uint8,
                                                     targets_uint8,
                                                     mean_training,
                                                     tuple_width_height_masks,
                                                     is_fully_connected)

def preprocess_context_portions_targets_numpy(portions_above_uint8, portions_left_uint8, targets_uint8,
                                              mean_training, tuple_width_height_masks, is_fully_connected):
    """Shifts the pixel intensity in each target patch and its two context portions, then masks the context portions.
    
    A mask covers the right side of the context portion
    located above the target patch. Its height is equal
    to `portions_above_uint8.shape[1]` and its width is
    equal to `tuple_width_height_masks[0]`. Another mask
    covers the bottom of the context portion located on
    the left side of the target patch. Its height is equal
    to `tuple_width_height_masks[1]` and its width is equal
    to `portions_left_uint8.shape[2]`.
    
    WARNING! The data-type and the shape of each array in
    the function arguments are not checked as these arrays
    come from `extract_context_portions_targets_from_channels_numpy`.
    
    Parameters
    ----------
    portions_above_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Context portions, each being located above a
        different target patch. `portions_above_uint8[i, :, :, :]`
        is the context portion located above the target
        patch of index i. `portions_above_uint8.shape[3]`
        is equal to 1.
    portions_left_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Context portions, each being located on the left
        side of a different target patch. `portions_left_uint8[i, :, :, :]`
        is the context portion located on the left side
        of the target patch of index i. `portions_left_uint8.shape[3]`
        is equal to 1.
    targets_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Target patches. `targets_uint8[i, :, :, :]` is
        the target patch of index i. `targets_uint8.shape[3]`
        is equal to 1.
    mean_training : float
        Mean pixels intensity computed over the same channel
        of different YCbCr images.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the width of the mask
        that covers the right side of the context portion located
        above the target patch. The 2nd integer in this tuple is
        the height of the mask that covers the bottom of the context
        portion located on the left side of the target patch.
    is_fully_connected : bool
        Is PNN fully-connected?
    
    Returns
    -------
    tuple
        If `is_fully_connected` is True,
            numpy.ndarray
                2D array with data-type `numpy.float32`.
                Flattened masked contexts.
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Target patches. The 4th array dimension
                is equal to 1.
        Otherwise,
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Masked context portions, each being located
                above a different target patch. The 4th array
                dimension is equal to 1.
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Masked context portions, each being located on
                the left side of a different target patch. The
                4th array dimension is equal to 1.
            numpy.ndarray
                4D array with data-type `numpy.float32`.
                Target patches.  The 4th array dimension is
                equal to 1.
    
    Raises
    ------
    ValueError
        If `tuple_width_height_masks[0]` does not belong
        to {0, 4, ..., `targets_uint8.shape[1]`}.
    ValueError
        If `tuple_width_height_masks[1]` does not belong
        to {0, 4, ..., `targets_uint8.shape[1]`}.
    
    """
    nb_targets = targets_uint8.shape[0]
    width_target = targets_uint8.shape[1]
    
    # If `len(tuple_width_height_masks)` is not equal to 2,
    # the unpacking below raises a `ValueError` exception.
    (width_mask_above, height_mask_left) = tuple_width_height_masks
    
    # In Numpy, wacky slicings do not raise any exception.
    if width_mask_above < 0 or width_mask_above > width_target or width_mask_above % 4 != 0:
        raise ValueError('`tuple_width_height_masks[0]` does not belong to {0, 4, ..., `targets_uint8.shape[1]`}.')
    if height_mask_left < 0 or height_mask_left > width_target or height_mask_left % 4 != 0:
        raise ValueError('`tuple_width_height_masks[1]` does not belong to {0, 4, ..., `targets_uint8.shape[1]`}.')
    
    # `portions_above_float32` contains the context portions,
    # each being located above a different target patch,
    # after subtracting `mean_training` and masking them.
    portions_above_float32 = portions_above_uint8.astype(numpy.float32) - mean_training
    portions_above_float32[:, :, 3*width_target - width_mask_above:, :] = 0.
    
    # `portions_left_float32` contains the context portions,
    # each being located on the left side of a different target
    # patch, after subtracting `mean_training` and masking them.
    portions_left_float32 = portions_left_uint8.astype(numpy.float32) - mean_training
    portions_left_float32[:, 2*width_target - height_mask_left:, :, :] = 0.
    
    # `targets_float32` contains the target patches after
    # subtracting `mean_training`.
    targets_float32 = targets_uint8.astype(numpy.float32) - mean_training
    if is_fully_connected:
        flattened_portions_above_float32 = numpy.reshape(portions_above_float32,
                                                         (nb_targets, -1))
        flattened_portions_left_float32 = numpy.reshape(portions_left_float32,
                                                        (nb_targets, -1))
        flattened_contexts_float32 = numpy.concatenate((flattened_portions_above_float32, flattened_portions_left_float32),
                                                       axis=1)
        return (flattened_contexts_float32, targets_float32)
    else:
        return (portions_above_float32, portions_left_float32, targets_float32)


