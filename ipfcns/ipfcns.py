"""A library containing functions for comparing Intra Prediction Fully-Connected Network (IPFCN-S) and PNN."""

import numpy

import tools.tools as tls

# The functions are sorted in alphabetic order.

def arrange_pair_groups_lines(group_lines_above_uint8, group_lines_left_uint8):
    """Arranges the group of reference lines located above a target patch and the group of reference lines located on the left side of this target patch in an image.
    
    Parameters
    ----------
    group_lines_above_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Group of reference lines located above a target
        patch. `group_lines_above_uint8.shape[2]` is equal
        to 1.
    group_lines_left_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Group of reference lines located on the left side
        of this target patch. `group_lines_left_uint8.shape[2]`
        is equal to 1.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image containing the two groups of reference
        lines. The 3rd array dimension is equal to 1.
    
    Raises
    ------
    TypeError
        If `group_lines_above_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        If `group_lines_left_uint8.dtype` is not equal to `numpy.uint8`.
    
    """
    if group_lines_above_uint8.dtype != numpy.uint8:
        raise TypeError('`group_lines_above_uint8.dtype` is not equal to `numpy.uint8`.')
    if group_lines_left_uint8.dtype != numpy.uint8:
        raise TypeError('`group_lines_left_uint8.dtype` is not equal to `numpy.uint8`.')
    width_image = group_lines_above_uint8.shape[1]
    image_uint8 = 255*numpy.ones((width_image, width_image, 1),
                                 dtype=numpy.uint8)
    
    # If `group_lines_above_uint8.shape[0]` is not equal
    # to 8, the copy below raises a `ValueError` exception.
    image_uint8[0:8, :, :] = group_lines_above_uint8
    
    # If `group_lines_left_uint8.shape` is not equal
    # to (`group_lines_above_uint8.shape[1] - 8`, 8, 1),
    # the copy below raises a `ValueError` exception.
    image_uint8[8:, 0:8, :] = group_lines_left_uint8
    return image_uint8

def arrange_flattened_pair_groups_lines(flattened_pair_groups_lines_uint8, width_target):
    """Arranges the flattened pair of reference lines groups in an image.
    
    Parameters
    ----------
    flattened_pair_groups_lines_uint8 : numpy.ndarray
        1D array with data-type `numpy.uint8`.
        Flattened pair of reference lines groups.
    width_target : int
        Width of the target patch.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image containing the flattened pair of reference
        lines groups after the arrangement. The 3rd array
        dimension is equal to 1.
    
    Raises
    ------
    TypeError
        If `flattened_pair_groups_lines_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `flattened_pair_groups_lines_uint8.ndim` is not equal to 1.
    
    """
    if flattened_pair_groups_lines_uint8.dtype != numpy.uint8:
        raise TypeError('`flattened_pair_groups_lines_uint8.dtype` is not equal to `numpy.uint8`.')
    if flattened_pair_groups_lines_uint8.ndim != 1:
        raise ValueError('`flattened_pair_groups_lines_uint8.ndim` is not equal to 1.')
    image_uint8 = 255*numpy.ones((2*width_target + 8, 2*width_target + 8, 1),
                                 dtype=numpy.uint8)
    image_uint8[0:8, :, :] = numpy.reshape(flattened_pair_groups_lines_uint8[0:16*width_target + 64],
                                           (8, 2*width_target + 8, 1))
    image_uint8[8:, 0:8, :] = numpy.reshape(flattened_pair_groups_lines_uint8[16*width_target + 64:],
                                            (2*width_target, 8, 1))
    return image_uint8

def extract_pair_groups_lines_from_channel(channel_single_or_pair_uint8, width_target, row_1st, col_1st):
    """Extracts the two groups of reference lines of a target patch from the image channel.
    
    Parameters
    ----------
    channel_single_or_pair_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 1, `channel_single_or_pair_uint8` is an image channel.
        If `channel_single_or_pair_uint8.shape[2]` is equal
        to 2, `channel_single_or_pair_uint8[:, :, 0:1]` is
        an image channel and `channel_single_or_pair_uint8[:, :, 1:2]`
        is this image channel with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1st : int
        Row of the 1st lines pixel in the image channel.
        The 1st lines pixel is the pixel at the top-left
        of the group of reference lines located above the
        target patch.
    col_1st : int
        Column of the 1st lines pixel in the image channel.
    
    Returns
    -------
    tuple
        numpy.ndarray
            3D array with data-type `numpy.uint8`.
            Group of reference lines located above the target patch.
            The array shape is equal to (8, `2*width_target + 8`,
            1). If `channel_single_or_pair_uint8.shape[2]` is equal
            to 1, the group of reference lines located above the
            target patch is extracted from the image channel. If
            `channel_single_or_pair_uint8.shape[2]` is equal to 2,
            it is extracted from the image channel with HEVC
            compression artifacts.
        numpy.ndarray
            3D array with data-type `numpy.uint8`.
            Group of reference lines located on the left side of the
            target patch. The array shape is equal to (`2*width_target`,
            8, 1). If `channel_single_or_pair_uint8.shape[2]` is equal
            to 1, the group of reference lines located on the left side
            of the target patch is extracted from the image channel. If
            `channel_single_or_pair_uint8.shape[2]` is equal to 2, it
            is extracted from the image channel with HEVC compression
            artifacts.
    
    Raises
    ------
    TypeError
        If `channel_single_or_pair_uint8.dtype` is not equal
        to `numpy.uint8`.
    ValueError
        If `width_target` is not positive.
    ValueError
        If `row_1st` is not positive.
    ValueError
        If `col_1st` is not positive.
    ValueError
        `row_1st + 2*width_target + 8` is not smaller than
        `channel_single_or_pair_uint8.shape[0]`.
    ValueError
        `col_1st + 2*width_target + 8` is not smaller than
        `channel_single_or_pair_uint8.shape[1]`.
    ValueError
        If `channel_single_or_pair_uint8.shape[2]` does not
        belong to {1, 2}.
    
    """
    if channel_single_or_pair_uint8.dtype != numpy.uint8:
        raise TypeError('`channel_single_or_pair_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If `channel_single_or_pair_uint8.ndim` is not equal
    # to 3, the unpacking below raises a `ValueError` exception.
    (height_channel, width_channel, nb_channels) = channel_single_or_pair_uint8.shape
    
    # If the five checks below did not exist, `extract_groups_reference_lines_from_channel`
    # would not crash if the slicings below go out of bounds.
    if width_target < 0:
        raise ValueError('`width_target` is not positive.')
    if row_1st < 0:
        raise ValueError('`row_1st` is not positive.')
    if col_1st < 0:
        raise ValueError('`col_1st` is not positive.')
    if row_1st + 2*width_target + 8 > height_channel:
        raise ValueError('`row_1st + 2*width_target + 8` is not smaller than `channel_single_or_pair_uint8.shape[0]`.')
    if col_1st + 2*width_target + 8 > width_channel:
        raise ValueError('`col_1st + 2*width_target + 8` is not smaller than `channel_single_or_pair_uint8.shape[1]`.')
    if nb_channels in (1, 2):
        i = nb_channels - 1
    else:
        raise ValueError('`channel_single_or_pair_uint8.shape[2]` does not belong to {1, 2}.')
    group_lines_above_uint8 = channel_single_or_pair_uint8[row_1st:row_1st + 8, col_1st:col_1st + 2*width_target + 8, i:i + 1]
    group_lines_left_uint8 = channel_single_or_pair_uint8[row_1st + 8:row_1st + 2*width_target + 8, col_1st:col_1st + 8, i:i + 1]
    return (group_lines_above_uint8, group_lines_left_uint8)

def extract_pairs_groups_lines_from_channel(channel_single_or_pair_uint8, width_target, row_1sts, col_1sts):
    """Extracts the two groups of reference lines of each target patch from the image channel.
    
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
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the
        1st pixel of the group of reference lines located above
        the target patch of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of
        the 1st pixel of the group of reference lines located
        above the target patch of index i.
    
    Returns
    -------
    tuple
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Groups of reference lines, each being located above
            a different target patch. The array shape is equal
            to (`row_1sts.size`, 8, `2*width_target + 8`, 1).
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Groups of reference lines, each being located on the
            left side of a different target patch. The array shape
            is equal to (`row_1sts.size`, `2*width_target`, 8, 1).
    
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
    # slicings below fire warnings instead of raising
    # exceptions.
    if not numpy.issubdtype(row_1sts.dtype, numpy.integer):
        raise TypeError('`row_1sts.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    if not numpy.issubdtype(col_1sts.dtype, numpy.integer):
        raise TypeError('`col_1sts.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    size_row_1sts = row_1sts.size
    
    # If the check below did not exist, `extract_pairs_groups_lines_from_channel`
    # would not crash if `col_1sts.size` is strictly larger than `row_1sts.size`.
    if col_1sts.size != size_row_1sts:
        raise ValueError('`col_1sts.size` is not equal to `row_1sts.size`.')
    groups_lines_above_uint8 = numpy.zeros((size_row_1sts, 8, 2*width_target + 8, 1), dtype=numpy.uint8)
    groups_lines_left_uint8 = numpy.zeros((size_row_1sts, 2*width_target, 8, 1), dtype=numpy.uint8)
    for i in range(size_row_1sts):
        
        # If `row_1sts.ndim` is not equal to 1, `row_1sts[i].item()`
        # raises a `ValueError` exception. Similarly, if `col_1sts.ndim`
        # is not equal to 1, `col_1sts[i].item()` raises a `ValueError`
        # exception.
        (groups_lines_above_uint8[i, :, :, :], groups_lines_left_uint8[i, :, :, :]) = \
            extract_pair_groups_lines_from_channel(channel_single_or_pair_uint8,
                                                   width_target,
                                                   row_1sts[i].item(),
                                                   col_1sts[i].item())
    return (groups_lines_above_uint8, groups_lines_left_uint8)

def extract_pairs_groups_lines_from_channels(channels_single_or_pair_uint8, width_target, row_1sts, col_1sts):
    """Extracts the two groups of reference lines of each target patch from the same channel of each image.
    
    Parameters
    ----------
    channels_single_or_pair_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. If `channels_single_or_pair_uint8.shape[3]`
        is equal to 1, `channels_single_or_pair_uint8[i, :, :, :]`
        is the channel of the image of index i. If `channels_single_or_pair_uint8.shape[3]`
        is equal to 2, `channels_single_or_pair_uint8[i, :, :, 0:1]`
        is the channel of the image of index i and `channels_single_or_pair_uint8[i, :, :, 1:2]`
        is the channel of the image of index i with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the
        1st pixel of the group of reference lines located above
        the target patch of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of
        the 1st pixel of the group of reference lines located
        above the target patch of index i.
    
    Returns
    -------
    tuple
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Groups of reference lines, each being located above
            a different target patch. The array shape is equal
            to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
            8, `2*width_target + 8`, 1).
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Groups of reference lines, each being located on the
            left side of a different target patch. The array shape
            is equal to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
            `2*width_target`, 8, 1).
    
    """
    nb_images = channels_single_or_pair_uint8.shape[0]
    size_row_1sts = row_1sts.size
    groups_lines_above_uint8 = numpy.zeros((nb_images*size_row_1sts, 8, 2*width_target + 8, 1),
                                           dtype=numpy.uint8)
    groups_lines_left_uint8 = numpy.zeros((nb_images*size_row_1sts, 2*width_target, 8, 1),
                                          dtype=numpy.uint8)
    for i in range(nb_images):
        (
            groups_lines_above_uint8[i*size_row_1sts:(i + 1)*size_row_1sts, :, :, :],
            groups_lines_left_uint8[i*size_row_1sts:(i + 1)*size_row_1sts, :, :, :]
        ) = extract_pairs_groups_lines_from_channel(channels_single_or_pair_uint8[i, :, :, :],
                                                    width_target,
                                                    row_1sts,
                                                    col_1sts)
    return (groups_lines_above_uint8, groups_lines_left_uint8)

def extract_pairs_groups_lines_from_channels_plus_preprocessing(channels_single_or_pair_uint8, width_target, row_1sts, col_1sts):
    """Extracts the two groups of reference lines of each target patch from the same channel of each image and preprocesses them.
    
    Parameters
    ----------
    channels_single_or_pair_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. If `channels_single_or_pair_uint8.shape[3]`
        is equal to 1, `channels_single_or_pair_uint8[i, :, :, :]`
        is the channel of the image of index i. If `channels_single_or_pair_uint8.shape[3]`
        is equal to 2, `channels_single_or_pair_uint8[i, :, :, 0:1]`
        is the channel of the image of index i and `channels_single_or_pair_uint8[i, :, :, 1:2]`
        is the channel of the image of index i with HEVC compression artifacts.
    width_target : int
        Width of the target patch.
    row_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_1sts[i]` is the row in the image channel of the
        1st pixel of the group of reference lines located above
        the target patch of index i.
    col_1sts : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_1sts[i]` is the column in the image channel of
        the 1st pixel of the group of reference lines located
        above the target patch of index i.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float32`.
            Flattened pairs of reference lines groups. The array shape
            is equal to (`row_1sts.size*channels_single_or_pair_uint8.shape[0]`,
            `64 + 32*width_target`).
        numpy.ndarray
            1D array with data-type `numpy.float32`.
            The array element of index i is the mean pixels intensity
            over the reference lines associated to the target patch of
            index i.
    
    """
    (groups_lines_above_uint8, groups_lines_left_uint8) = \
        extract_pairs_groups_lines_from_channels(channels_single_or_pair_uint8,
                                                 width_target,
                                                 row_1sts,
                                                 col_1sts)
    return preprocess_pairs_groups_lines(groups_lines_above_uint8,
                                         groups_lines_left_uint8)

def predict_by_batch_via_ipfcns(flattened_pairs_groups_lines_float32, net_ipfcns, width_target, batch_size):
    """Computes a prediction of each target patch via a IPFCN-S, one batch at a time.
    
    Parameters
    ----------
    flattened_pairs_groups_lines_float32 : numpy.ndarray
        2D array with data-type `numpy.float32`.
        Flattened pairs of reference lines groups. `flattened_pairs_groups_lines_float32[i, :]`
        contain the flattened pair of reference lines
        groups associated to the target patch of index i.
    net_ipfcns : Net (class in Caffe)
        IPFCN-S instance.
    width_target : int
        Width of the target patch.
    batch_size : int
        Batch size.
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.float32`.
        Prediction of the target patches. The 4th
        array dimension is equal to 1.
    
    """
    nb_predictions = flattened_pairs_groups_lines_float32.shape[0]
    nb_batches = tls.divide_ints_check_divisible(nb_predictions,
                                                 batch_size)
    predictions_float32 = numpy.zeros((nb_predictions, width_target, width_target, 1),
                                      dtype=numpy.float32)
    for i in range(nb_batches):
        
        # `net_ipfcns.blobs['data'].data[...].dtype` is equal to `numpy.float32`.
        net_ipfcns.blobs['data'].data[...] = numpy.expand_dims(
            numpy.expand_dims(flattened_pairs_groups_lines_float32[i*batch_size:(i + 1)*batch_size, :], 2),
            3
        )
        dict_out = net_ipfcns.forward()
        predictions_float32[i*batch_size:(i + 1)*batch_size, :, :, :] = numpy.reshape(
            dict_out['fc4'],
            (batch_size, width_target, width_target, 1)
        )
    return predictions_float32

def preprocess_pairs_groups_lines(groups_lines_above_uint8, groups_lines_left_uint8):
    """Shifts the pixels intensity in each group of reference lines.
    
    WARNING! The data-type and the shape of each array in the
    function arguments are not checked because these arrays usually
    come from `extract_pairs_groups_lines_from_channels`.
    
    Parameters
    ----------
    groups_lines_above_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Groups of reference lines, each being located
        above a different target patch. `groups_lines_above_uint8[i, :, :, :]`
        is the group of reference lines located above
        the target patch of index i. `groups_lines_above_uint8.shape[3]`
        is equal to 1.
    groups_lines_left_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Groups of reference lines, each being located
        on the left side of a different target patch.
        `groups_lines_left_uint8[i, :, :, :]` is the
        group of reference lines located on the left
        side of the target patch of index i. `groups_lines_left_uint8.shape[3]`
        is equal to 1.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float32`.
            Flattened pairs of reference lines groups.
        numpy.ndarray
            1D array with data-type `numpy.float32`.
            The ith array element is the mean pixels intensity
            over the two groups of reference lines associated
            to the target patch of index i.
    
    """
    groups_lines_above_float32 = groups_lines_above_uint8.astype(numpy.float32)
    groups_lines_left_float32 = groups_lines_left_uint8.astype(numpy.float32)
    size_group = numpy.prod(groups_lines_above_float32.shape[1:]) + numpy.prod(groups_lines_left_uint8.shape[1:])
    
    # `means_float32[i]` is the mean pixels intensity over the
    # grouping of `groups_lines_above_float32[i, :, :, :]` and
    # `groups_lines_left_float32[i, :, :, :]`.
    means_float32 = (numpy.sum(groups_lines_above_float32, axis=(1, 2, 3)) + numpy.sum(groups_lines_left_float32, axis=(1, 2, 3)))/size_group
    expanded_means_float32 = numpy.expand_dims(
        numpy.expand_dims(numpy.expand_dims(means_float32, 1), 2),
        3
    )
    groups_lines_above_float32 -= numpy.tile(expanded_means_float32,
                                             (1, groups_lines_above_float32.shape[1], groups_lines_above_float32.shape[2], 1))
    groups_lines_left_float32 -= numpy.tile(expanded_means_float32,
                                            (1, groups_lines_left_float32.shape[1], groups_lines_left_float32.shape[2], 1))
    flattened_groups_lines_above_float32 = numpy.reshape(groups_lines_above_float32,
                                                         (groups_lines_above_float32.shape[0], -1))
    flattened_groups_lines_left_float32 = numpy.reshape(groups_lines_left_float32,
                                                        (groups_lines_left_float32.shape[0], -1))
    flattened_pairs_groups_lines_float32 = numpy.concatenate(
        (flattened_groups_lines_above_float32, flattened_groups_lines_left_float32),
        axis=1
    )
    return (flattened_pairs_groups_lines_float32, means_float32)

def visualize_flattened_pair_groups_lines(flattened_pair_groups_lines_float32, width_target, mean_pair_groups_lines,
                                          path, coefficient_enlargement=None):
    """Arranges the flattened pair of reference lines groups in an image and saves the image.
    
    Parameters
    ----------
    flattened_pair_groups_lines_float32 : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Flattened pair of reference lines groups.
    width_target : int
        Width of the target patch.
    mean_pair_groups_lines : float
        Mean pixels intensity over the pair of
        reference lines groups.
    path : str
        Path to the saved image. The path ends
        with ".png".
    coefficient_enlargement : int, optional
        Coefficient for enlarging the image along
        its first two dimensions before saving it.
        The default value is None, meaning that
        there is no enlargement of the image.
    
    """
    flattened_pair_groups_lines_uint8 = tls.cast_float_to_uint8(flattened_pair_groups_lines_float32 + mean_pair_groups_lines)
    image_uint8 = arrange_flattened_pair_groups_lines(flattened_pair_groups_lines_uint8,
                                                      width_target)
    tls.save_image(path,
                   numpy.squeeze(image_uint8, axis=2),
                   coefficient_enlargement=coefficient_enlargement)


