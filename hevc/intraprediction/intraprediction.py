"""A library containing functions for simulating the intra prediction in HEVC."""

import numpy

import hevc.intraprediction.interface
import tools.tools as tls

# The functions are sorted in alphabetic order.

def extract_intra_pattern(channel_uint8, width_target, row_ref, col_ref, tuple_width_height_masks):
    """Extracts the intra pattern of a target patch from the image channel.
    
    Parameters
    ----------
    channel_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Image channel. `channel_uint8.shape[2]` is equal to 1.
    width_target : int
        Width of the target patch.
    row_ref : int
        Row of the 1st intra pattern pixel in the
        image channel. Note that the row of the 1st
        target patch pixel in the image channel is
        equal to `row_ref + 1`.
    col_ref : int
        Column of the 1st intra pattern pixel in the
        image channel. Note that the column of the 1st
        target patch pixel in the image channel is
        equal to `col_ref + 1`.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the
        width of the unknown part at the right
        side of the intra pattern portion located
        above the target patch. The 2nd integer
        in this tuple is the height of the unknown
        part at the the bottom of the intra pattern
        portion located on the left side of the
        target patch.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Intra pattern of the target patch. The array
        shape is equal to (`2*width_target + 1 - tuple_width_height_masks[1]`,
        `2*width_target + 1 - tuple_width_height_masks[0]`, 1).
    
    Raises
    ------
    TypeError
        If `channel_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `channel_uint8.ndim` is not equal to 3.
    ValueError
        If `width_target` is not positive.
    ValueError
        If `row_ref` is not positive.
    ValueError
        If `col_ref` is not positive.
    ValueError
        If `tuple_width_height_masks[0]` does not belong
        to {0, 4, ..., `width_target`}.
    ValueError
        If `tuple_width_height_masks[1]` does not belong
        to {0, 4, ..., `width_target`}.
    
    """
    if channel_uint8.dtype != numpy.uint8:
        raise TypeError('`channel_uint8.dtype` is not equal to `numpy.uint8`.')
    if channel_uint8.ndim != 3:
        raise ValueError('`channel_uint8.ndim` is not equal to 3.')
    if width_target < 0:
        raise ValueError('`width_target` is not positive.')
    if row_ref < 0:
        raise ValueError('`row_ref` is not positive.')
    if col_ref < 0:
        raise ValueError('`col_ref` is not positive.')
    
    # If `len(tuple_width_height_masks)` is not equal to 2,
    # the unpacking below raises a `ValueError` exception.
    (width_mask_above, height_mask_left) = tuple_width_height_masks
    if width_mask_above < 0 or width_mask_above > width_target or width_mask_above % 4 != 0:
        raise ValueError('`tuple_width_height_masks[0]` does not belong to {0, 4, ..., `width_target`}.')
    if height_mask_left < 0 or height_mask_left > width_target or height_mask_left % 4 != 0:
        raise ValueError('`tuple_width_height_masks[1]` does not belong to {0, 4, ..., `width_target`}.')
    height_pattern = 2*width_target + 1 - height_mask_left
    width_pattern = 2*width_target + 1 - width_mask_above
    
    # The area in `intra_pattern_uint8` that does not belong
    # to the intra pattern is colored in white for display.
    intra_pattern_uint8 = 255*numpy.ones((height_pattern, width_pattern, 1),
                                         dtype=numpy.uint8)
    
    # If `row_ref + height_pattern` is strictly larger than
    # `channel_uint8.shape[0]`, the copy below raises a `ValueError`
    # exception.  If `col_ref + width_pattern` is strictly larger
    # than `channel_uint8.shape[1]`, the copy below raises
    # a `ValueError` exception.
    intra_pattern_uint8[:, 0, :] = channel_uint8[row_ref:row_ref + height_pattern, col_ref, :]
    intra_pattern_uint8[0, :, :] = channel_uint8[row_ref, col_ref:col_ref + width_pattern, :]
    return intra_pattern_uint8

def extract_intra_patterns(channels_uint8, width_target, row_refs, col_refs, tuple_width_height_masks):
    """Extracts several intra patterns of target patches from the same channel of each image.
    
    Parameters
    ----------
    channels_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Images channel. `channels_uint8[i, :, :, :]` is the
        channel of the image of index i. `channels_uint8.shape[3]`
        is equal to 1.
    width_target : int
        Width of the target patch.
    row_refs : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `row_refs[i]` is the row of the 1st pixel of the
        intra pattern of index i in the image channel.
    col_refs : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        `col_refs[i]` is the column of the 1st pixel of the
        intra pattern of index i in the image channel.
    tuple_width_height_masks : tuple
        The 1st integer in this tuple is the
        width of the unknown part at the right
        side of the intra pattern portion located
        above the target patch. The 2nd integer
        in this tuple is the height of the unknown
        part at the the bottom of the intra pattern
        portion located on the left side of the
        target patch.
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Intra patterns of target patches. The array shape
        is equal to (`row_refs.size*channels_uint8.shape[0]`,
        `2*width_target + 1 - tuple_width_height_masks[1]`,
        `2*width_target + 1 - tuple_width_height_masks[0]`, 1).
    
    Raises
    ------
    TypeError
        If `row_refs.dtype` is not smaller than `numpy.integer`
        in type hierarchy.
    TypeError
        If `col_refs.dtype` is not smaller than `numpy.integer`
        in type hierarchy.
    ValueError
        If `col_refs.size` is not equal to `row_refs.size`.
    
    """
    # The check below is crucial because, if the elements
    # of `row_refs` are floats or booleans for instance, the
    # slicings below fire warnings instead of raising
    # exceptions.
    if not numpy.issubdtype(row_refs.dtype, numpy.integer):
        raise TypeError('`row_refs.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    if not numpy.issubdtype(col_refs.dtype, numpy.integer):
        raise TypeError('`col_refs.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    size_row_refs = row_refs.size
    
    # If the check below did not exist, `extract_intra_patterns` would
    # not crash if `col_refs.size` is strictly larger than `row_refs.size`.
    if col_refs.size != size_row_refs:
        raise ValueError('`col_refs.size` is not equal to `row_refs.size`.')
    height_pattern = 2*width_target + 1 - tuple_width_height_masks[1]
    width_pattern = 2*width_target + 1 - tuple_width_height_masks[0]
    intra_patterns_uint8 = numpy.zeros((channels_uint8.shape[0]*size_row_refs, height_pattern, width_pattern, 1),
                                       dtype=numpy.uint8)
    for i in range(channels_uint8.shape[0]):
        for j in range(size_row_refs):
            intra_patterns_uint8[i*size_row_refs + j, :, :, :] = extract_intra_pattern(channels_uint8[i, :, :, :],
                                                                                       width_target,
                                                                                       row_refs[j].item(),
                                                                                       col_refs[j].item(),
                                                                                       tuple_width_height_masks)
    return intra_patterns_uint8

def predict_series_via_hevc_best_mode(intra_patterns_uint8, targets_uint8):
    """Computes a prediction of each target patch via the best HEVC intra prediction mode in terms of prediction PSNR.
    
    Parameters
    ----------
    intra_patterns_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Intra patterns. `intra_patterns_uint8[i, :, :, :]`
        is the intra pattern of the target patch of index i.
        `intra_patterns_uint8.shape[3]` is equal to 1.
    targets_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Target patches. `targets_uint8[i, :, :, :]` is the
        target patch of index i. `targets_uint8.shape[3]` is
        equal to 1.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.uint8`.
            The element of index i in this array is the index of the
            best HEVC intra prediction mode in terms of prediction
            PSNR for the target patch of index i.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            The element of index i in this array is the PSNR between the
            target patch of index i and its prediction via the best HEVC
            intra prediction mode in terms of prediction PSNR.
        numpy.ndarray
            4D array with data-type `numpy.uint8`.
            Prediction of each target patch via the best HEVC
            intra prediction mode in terms of prediction PSNR.
            The 4th array dimension is equal to 1.
    
    """
    nb_targets = targets_uint8.shape[0]
    indices_hevc_best_mode = numpy.zeros(nb_targets,
                                         dtype=numpy.uint8)
    psnrs_hevc_best_mode = numpy.zeros(nb_targets)
    predictions_hevc_best_mode_uint8 = numpy.zeros(targets_uint8.shape,
                                                   dtype=numpy.uint8)
    for i in range(nb_targets):
        (indices_hevc_best_mode[i], psnrs_hevc_best_mode[i], predictions_hevc_best_mode_uint8[i, :, :, :]) = \
            predict_via_hevc_best_mode(intra_patterns_uint8[i, :, :, :],
                                       targets_uint8[i, :, :, :])
    return (indices_hevc_best_mode, psnrs_hevc_best_mode, predictions_hevc_best_mode_uint8)

def predict_via_hevc_best_mode(intra_pattern_uint8, target_uint8):
    """Computes a prediction of the target patch via the best HEVC intra prediction mode in terms of prediction PSNR.
    
    Parameters
    ----------
    intra_pattern_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Intra pattern of the target patch. `intra_pattern_uint8.shape[2]`
        is equal to 1.
    target_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Target patch. `target_uint8.shape[2]` is equal to 1.
    
    Returns
    -------
    tuple
        int
            Index of the best HEVC intra prediction mode
            in terms of prediction PSNR.
        numpy.float64
            PSNR between the target patch and its prediction
            via the best HEVC intra prediction mode in terms
            of prediction PSNR.
        numpy.ndarray
            3D array with data-type `numpy.uint8`.
            Prediction of the target patch via the best HEVC
            intra prediction mode in terms of prediction PSNR.
            The 3rd array dimension is equal to 1.
    
    """
    width_target = target_uint8.shape[0]
    squeezed_target_uint8 = numpy.squeeze(target_uint8,
                                          axis=2)
    index_hevc_best_mode = 0
    psnr_hevc_best_mode = numpy.float64(0.)
    prediction_hevc_best_mode_uint8 = numpy.zeros((width_target, width_target, 1),
                                                  dtype=numpy.uint8)
    
    # The method `numpy.ndarray.copy` returns
    # a C-contiguous copy of the array by default.
    intra_pattern_c_contiguous_uint8 = intra_pattern_uint8.copy()
    for index_mode in range(35):
        
        # `hevc.intraprediction.interface.predict_via_hevc_mode`
        # checks that `intra_pattern_c_contiguous_uint8.dtype` is equal
        # to `numpy.uint8` and `intra_pattern_c_contiguous_uint8.ndim`
        # is equal to 2.
        prediction_mode_uint8 = hevc.intraprediction.interface.predict_via_hevc_mode(intra_pattern_c_contiguous_uint8,
                                                                                     width_target,
                                                                                     index_mode)
        squeezed_prediction_mode_uint8 = numpy.squeeze(prediction_mode_uint8,
                                                       axis=2)
        
        # `tls.compute_psnr` checks that `target_uint8.dtype`
        # is equal to `numpy.uint8`.
        psnr_mode = tls.compute_psnr(squeezed_target_uint8,
                                     squeezed_prediction_mode_uint8)
        if psnr_mode > psnr_hevc_best_mode:
            index_hevc_best_mode = index_mode
            psnr_hevc_best_mode = psnr_mode
            prediction_hevc_best_mode_uint8 = prediction_mode_uint8
    return (index_hevc_best_mode, psnr_hevc_best_mode, prediction_hevc_best_mode_uint8)


