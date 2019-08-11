"""A library defining a Cython interface for running the HEVC intra prediction."""

import numpy
cimport numpy

cdef extern from "c++/source/extracted_hevc_intraprediction.h":
    cdef void hevc_intraprediction(const int&,
                                   const int&,
                                   const int&,
                                   const numpy.uint8_t*,
                                   numpy.uint8_t*,
                                   const unsigned int&) except +

def predict_via_hevc_mode(numpy.ndarray[numpy.uint8_t, ndim=3] intra_pattern_uint8,
                          int width_target,
                          unsigned int index_mode):
    """Computes a prediction of the target patch via a given HEVC intra prediction mode.
    
    Parameters
    ----------
    intra_pattern_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Intra pattern of the target patch. `intra_pattern_uint8.shape[2]`
        is equal to 1.
    width_target : int
        Width of the target patch.
    index_mode : int
        Index of the given HEVC intra prediction mode.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Prediction of the target patch via the given
        HEVC intra prediction mode. The 3rd array
        dimension is equal to 1.
    
    Raises
    ------
    ValueError
        If `intra_pattern_uint8` is not C-contiguous.
    ValueError
        If `intra_pattern_uint8.shape[2]` is not equal to 1.
    
    """
    # Cython automatically checks the number of dimensions
    # and the data-type of `intra_pattern_uint8`.
    if not intra_pattern_uint8.flags.c_contiguous:
        raise ValueError('`intra_pattern_uint8` is not C-contiguous.')
    if intra_pattern_uint8.shape[2] != 1:
        raise ValueError('`intra_pattern_uint8.shape[2]` is not equal to 1.')
    cdef int height_intra_pattern = intra_pattern_uint8.shape[0]
    cdef int width_intra_pattern = intra_pattern_uint8.shape[1]
    cdef numpy.ndarray[numpy.uint8_t, ndim=3] prediction_uint8 = numpy.zeros((width_target, width_target, 1),
                                                                             dtype=numpy.uint8)
    hevc_intraprediction(height_intra_pattern,
                         width_intra_pattern,
                         width_target,
                         &intra_pattern_uint8[0, 0, 0],
                         &prediction_uint8[0, 0, 0],
                         index_mode)
    return prediction_uint8


