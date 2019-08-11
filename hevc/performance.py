"""A library containing functions for evaluating the performance of HEVC in terms of rate-distortion."""

import numpy
import os
import re

import hevc.running
import tools.tools as tls

# The functions are sorted in alphabetic order.

def collect_computation_time(path_to_log_encoder_or_decoder):
    """Collects the computation time in the log file of either the HEVC encoder or the HEVC decoder.
    
    Parameters
    ----------
    path_to_log_encoder_or_decoder : str
        Path to the log file of either the HEVC encoder
        or the HEVC decoder.
    
    Returns
    -------
    float
        Computation time.
    
    Raises
    ------
    IOError
        If the computation time is not found in the file
        at `path_to_log_encoder_or_decoder`.
    
    """
    pattern = '(?<=Total Time:)\s*[0-9]*\.[0-9]*\s*(?=sec)'
    with open(path_to_log_encoder_or_decoder, 'r') as file:
        for line_text in file:
            
            # `re.search` scans `line_text`, looking for the first
            # location where the regular expression `pattern` produces
            # a match.
            sre_match = re.search(pattern,
                                  line_text)
            if sre_match is None:
                continue
            else:
                return float(sre_match.group(0))
    raise IOError('The computation time is not found in the file at "{}".'.format(path_to_log_encoder_or_decoder))

def collect_nb_bits_psnrs_from_log_encoder(path_to_log_encoder):
    """Collects the number of bits and the three PSNRS in the HEVC encoder log file.
    
    Parameters
    ----------
    path_to_log_encoder : str
        Path to the HEVC encoder log file. The path
        ends with ".txt".
    
    Returns
    -------
    tuple
        int
            Number of bits in the bitstream.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Each array element is the PSNR between a channel
            and its reconstruction after the decoding via HEVC.
    
    Raises
    ------
    IOError
        If "POC" does not exist in the HEVC encoder log file.
    
    """
    with open(path_to_log_encoder, 'r') as file:
        for line_text in file:
            
            # The text line starting with "POC" contains
            # the number of bits and the three PSNRS.
            index_poc = line_text.find('POC')
            if index_poc == -1:
                continue
            else:
                pattern_bits = '(?<=\))\s*[0-9]*\s*(?=bits)'
                sre_match_bits = re.search(pattern_bits,
                                           line_text)
                
                # If `sre_match_bits` is None, an `AttributeError`
                # exception is raised.
                nb_bits = int(sre_match_bits.group(0))
                pattern_psnrs = '(?<={})\s*[0-9]*\.[0-9]*\s*(?=dB)'
                sre_match_psnr_y = re.search(pattern_psnrs.format('Y'),
                                             line_text)
                sre_match_psnr_u = re.search(pattern_psnrs.format('U'),
                                             line_text)
                sre_match_psnr_v = re.search(pattern_psnrs.format('V'),
                                             line_text)
                psnrs = numpy.zeros(3)
                
                # If either `sre_match_psnr_y` or `sre_match_psnr_u` or
                # `sre_match_psnr_v` is None, an `AttributeError` is raised.
                psnrs[0] = float(sre_match_psnr_y.group(0))
                psnrs[1] = float(sre_match_psnr_u.group(0))
                psnrs[2] = float(sre_match_psnr_v.group(0))
                return (nb_bits, psnrs)
    raise IOError('"POC" does not exist in the HEVC encoder log file.')

def compute_rate_psnr(image_before_encoding_hevc_uint8or16, path_to_before_encoding_hevc, path_to_after_encoding_hevc,
                      path_to_after_decoding_hevc, path_to_cfg, path_to_bitstream, path_to_exe_encoder, path_to_exe_decoder,
                      path_to_log_encoder, path_to_log_decoder, qp, dict_substitution_switch):
    """Computes the rate and the PSNR of the encoding and decoding of the image via HEVC at a single quantization parameter.
    
    The computation time of the encoding of the image
    via HEVC at this quantization parameter and the
    computation time of its decoding are collected.
    
    Parameters
    ----------
    image_before_encoding_hevc_uint8or16 : numpy.ndarray
        3D array with data-type either `numpy.uint8` or `numpy.uint16`.
        Image before the encoding via HEVC. If `image_before_encoding_hevc_uint8or16.shape[2]`
        is equal to 1, `image_before_encoding_hevc_uint8or16.shape[2]` is
        considered as a luminance image. If `image_before_encoding_hevc_uint8or16.shape[2]`
        is equal to 3, `image_before_encoding_hevc_uint8or16.shape[2]` is
        considered as a YCbCr image.
    path_to_before_encoding_hevc : str
        Path to the file storing the image before the encoding
        via HEVC. `path_to_before_encoding_hevc` ends with ".yuv".
    path_to_after_encoding_hevc : str
        Path to the file storing the reconstructed image after
        the encoding via HEVC. `path_to_after_encoding_hevc`
        ends with ".yuv".
    path_to_after_decoding_hevc : str
        Path to the file storing the reconstructed image after
        the decoding via HEVC. `path_to_after_decoding_hevc`
        ends with ".yuv".
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    path_to_bitstream : str
        Path to the bitstream file. `path_to_bitstream` ends
        with ".bin".
    path_to_exe_encoder : str
        Path to the executable of the HEVC encoder.
    path_to_exe_decoder : str
        Path to the executable of the HEVC decoder.
    path_to_log_encoder : str
        Path to the HEVC encoder log file. The path
        ends with ".txt".
    path_to_log_decoder : str
        Path to the HEVC decoder log file. The path
        ends with ".txt".
    qp : int
        Quantization parameter.
    dict_substitution_switch : either dict or None
        path_to_additional_directory : str
            Path to the directory containing the file "loading.py".
        path_to_mean_training : str
            Path to the file containing the mean pixels luminance
            computed over different luminance images. The path ends
            with ".pkl".
        path_to_file_paths_to_graphs_output : str
            Path to the file containing the path to the output graph
            of each PNN used inside HEVC. The path ends with ".txt".
        If `dict_substitution_switch` is None, `path_to_exe_encoder`
        is the path to neither to the encoder of HEVC with the substitution
        nor the encoder of HEVC with the switch.
    
    Returns
    -------
    tuple
        numpy.ndarray
            Array with the same data-type and the same shape as
            `image_before_encoding_hevc_uint8or16`.
            Reconstructed image after the decoding via HEVC.
        float
            Rate of the encoding and decoding of the image
            via HEVC.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            PSNRs of the encoding and decoding of the image
            via HEVC. The first array element is the PSNR for
            the luminance channel. The last two array elements
            are the PSNRs for the blue and red chrominance channels
            respectively. If a luminance image is encoded and
            decoded via HEVC, the PSNRs for the blue and red
            chrominance channels are equal to 0.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Computation time of the encoding of the image via HEVC
            at this quantization parameter and the computation time
            of its decoding.
    
    """
    reconstructed_image_after_decoding_hevc_uint8or16 = \
        hevc.running.encode_decode_image(image_before_encoding_hevc_uint8or16,
                                         path_to_before_encoding_hevc,
                                         path_to_after_encoding_hevc,
                                         path_to_after_decoding_hevc,
                                         path_to_cfg,
                                         path_to_bitstream,
                                         path_to_exe_encoder,
                                         path_to_exe_decoder,
                                         path_to_log_encoder,
                                         path_to_log_decoder,
                                         qp,
                                         dict_substitution_switch)
    (nb_bits, psnrs) = collect_nb_bits_psnrs_from_log_encoder(path_to_log_encoder)
    computation_times = numpy.zeros(2)
    computation_times[0] = collect_computation_time(path_to_log_encoder)
    computation_times[1] = collect_computation_time(path_to_log_decoder)
    
    # The two log files are no longer needed.
    os.remove(path_to_log_encoder)
    os.remove(path_to_log_decoder)
    rate = convert_nb_bits_into_rate(nb_bits,
                                     image_before_encoding_hevc_uint8or16.shape)
    return (reconstructed_image_after_decoding_hevc_uint8or16, rate, psnrs, computation_times)

def compute_rates_psnrs(images_before_encoding_hevc_uint8or16, path_to_before_encoding_hevc, path_to_after_encoding_hevc,
                        path_to_after_decoding_hevc, path_to_cfg, path_to_bitstream, path_to_exe_encoder, path_to_exe_decoder,
                        path_to_log_encoder, path_to_log_decoder, qps_int, dict_substitution_switch, dict_visualization):
    """Computes the rate and the PSNR of the encoding and decoding of each image via HEVC for each quantization parameter.
    
    The computation time of the encoding of each image
    via HEVC at each quantization parameter and the
    computation time of its decoding are collected.
    
    Parameters
    ----------
    images_before_encoding_hevc_uint8or16 : numpy.ndarray
        4D array with data-type either `numpy.uint8` or `numpy.uint16`.
        Images before the encoding via HEVC. `images_before_encoding_hevc_uint8or16[i, :, :, :]`
        is the image of index i. If `images_before_encoding_hevc_uint8or16.shape[3]`
        is equal to 1, the images are viewed as luminance images. If
        `images_before_encoding_hevc_uint8or16.shape[3]` is equal to
        3, the images are viewed as YCbCr images.
    path_to_before_encoding_hevc : str
        Path to the file storing the image before the encoding
        via HEVC. `path_to_before_encoding_hevc` ends with ".yuv".
    path_to_after_encoding_hevc : str
        Path to the file storing the reconstructed image after
        the encoding via HEVC. `path_to_after_encoding_hevc`
        ends with ".yuv".
    path_to_after_decoding_hevc : str
        Path to the file storing the reconstructed image after
        the decoding via HEVC. `path_to_after_decoding_hevc`
        ends with ".yuv".
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    path_to_bitstream : str
        Path to a bitstream file. `path_to_bitstream` ends
        with ".bin".
    path_to_exe_encoder : str
        Path to the executable of the HEVC encoder.
    path_to_exe_decoder : str
        Path to the executable of the HEVC decoder.
    path_to_log_encoder : str
        Path to the HEVC encoder log file. The path
        ends with ".txt".
    path_to_log_decoder : str
        Path to the HEVC decoder log file. The path
        ends with ".txt".
    qps_int : numpy.ndarray
        1D array whose data-type is smaller than `numpy.integer`
        in type hierarchy.
        Quantization parameters.
    dict_substitution_switch : either dict or None
        path_to_additional_directory : str
            Path to the directory containing the file "loading.py".
        path_to_mean_training : str
            Path to the file containing the mean pixels luminance
            computed over different luminance images. The path ends
            with ".pkl".
        path_to_file_paths_to_graphs_output : str
            Path to the file containing the path to the output graph
            of each PNN used inside HEVC. The path ends with ".txt".
        If `dict_substitution_switch` is None, `path_to_exe_encoder`
        is the path to neither to the encoder of HEVC with the substitution
        nor the encoder of HEVC with the switch.
    dict_visualization : either dict or None
        rows_columns_top_left : numpy.ndarray
            2D array whose data-type is smaller than
            `numpy.integer` in type hierarchy.
            `rows_columns_top_left[:, i]` contains the
            row and the column of the image pixel at the
            top-left of the crop of index i.
        width_crop : int
            Width of the crop.
        list_indices_rotation : list
            Each integer in this list is the index of a
            rotated image.
        paths_to_saved_rotated_images_crops : list
            `paths_to_saved_rotated_images_crops` is a list of
            lists of lists of lists of paths. `paths_to_saved_rotated_images_crops[i][j]`
            contains the paths to saved visualizations for the
            reconstructed image of index j after the decoding
            via HEVC at the quantization parameter of index i.
        If `dict_visualization` is None, no visualization is saved.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the rate of the encoding and decoding
            of the image of index j via HEVC at the quantization
            parameter of index i.
        numpy.ndarray
            3D array with data-type `numpy.float64`.
            The element at the position [i, j, k] in this
            array is the PSNR of the encoding and decoding
            of the channel of index k of the image of index
            j via HEVC at the quantization parameter of index i.
        numpy.ndarray
            3D array with data-type `numpy.float64`.
            The two elements at the position [i, j, :] in
            this array are the computation time of the encoding
            of the image of index j via HEVC at the quantization
            parameter of index i and the computation time of its
            decoding.
    
    """
    nb_qps = qps_int.size
    nb_images = images_before_encoding_hevc_uint8or16.shape[0]
    rates = numpy.zeros((nb_qps, nb_images))
    psnrs = numpy.zeros((nb_qps, nb_images, 3))
    computation_times = numpy.zeros((nb_qps, nb_images, 2))
    for i in range(nb_qps):
        qp = qps_int[i].item()
        for j in range(nb_images):
            (reconstructed_image_after_decoding_hevc_uint8or16, rates[i, j], psnrs[i, j, :], computation_times[i, j, :]) = \
                compute_rate_psnr(images_before_encoding_hevc_uint8or16[j, :, :, :],
                                  path_to_before_encoding_hevc,
                                  path_to_after_encoding_hevc,
                                  path_to_after_decoding_hevc,
                                  path_to_cfg,
                                  path_to_bitstream,
                                  path_to_exe_encoder,
                                  path_to_exe_decoder,
                                  path_to_log_encoder,
                                  path_to_log_decoder,
                                  qp,
                                  dict_substitution_switch)
            
            # If `reconstructed_image_after_decoding_hevc_uint8or16.dtype`
            # is not equal to `numpy.uint8`, no visualization is saved.
            if dict_visualization is not None and reconstructed_image_after_decoding_hevc_uint8or16.dtype == numpy.uint8:
                tls.visualize_rotated_image(reconstructed_image_after_decoding_hevc_uint8or16,
                                            dict_visualization['rows_columns_top_left'],
                                            dict_visualization['width_crop'],
                                            j in dict_visualization['list_indices_rotation'],
                                            dict_visualization['paths_to_saved_rotated_images_crops'][i][j])
    
    # If luminance images are encoded and decoded via
    # HEVC, all the PSNRs for the blue and red chrominance
    # channels have to be equal to 0.
    if images_before_encoding_hevc_uint8or16.shape[3] == 1:
        are_zero_slices = numpy.array([False, True, True])
        psnrs = tls.check_remove_zero_slices_in_array_3d(psnrs,
                                                         are_zero_slices)
    return (rates, psnrs, computation_times)

def convert_nb_bits_into_rate(nb_bits, shape_image):
    """Converts the number of bits in the bitstream to a rate.
    
    Parameters
    ----------
    nb_bits : int
        Number of bits in the bitstream.
    shape_image : tuple
        Shape of the encoded and decoded image.
    
    Returns
    -------
    float
        Rate of the encoding and decoding of the image.
    
    Raises
    ------
    ValueError
        If `shape_image[2]` does not belong to {1, 3}.
    
    """
    # If `len(shape_image)` is not equal to 3, the unpacking
    # below raises a `ValueError` exception.
    (height_image, width_image, nb_channels) = shape_image
    
    # If the 3rd dimension is equal to 1, a luminance
    # image is considered. If it is equal to 3, a YCbCr
    # image in 4:2:0 is considered.
    if nb_channels == 1:
        nb_pixels = height_image*width_image
    elif nb_channels == 3:
        nb_pixels = 3*height_image*width_image//2
    else:
        raise ValueError('`shape_image[2]` does not belong to {1, 3}.')
    return float(nb_bits)/nb_pixels


