"""A library containing functions for running HEVC."""

import numpy
import os
import subprocess

# The functions are sorted in alphabetic order.

def encode_decode_image(image_before_encoding_hevc_uint8or16, path_to_before_encoding_hevc, path_to_after_encoding_hevc,
                        path_to_after_decoding_hevc, path_to_cfg, path_to_bitstream, path_to_exe_encoder, path_to_exe_decoder,
                        path_to_log_encoder, path_to_log_decoder, qp, dict_substitution_switch):
    """Encodes and decodes the image via HEVC.
    
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
    numpy.ndarray
        Array with the same data-type and the same shape as
        `image_before_encoding_hevc_uint8or16`.
        Reconstructed image after the decoding via HEVC.
    
    """
    write_400_or_420(numpy.expand_dims(image_before_encoding_hevc_uint8or16, 3),
                     path_to_before_encoding_hevc)
    
    # If `image_before_encoding_hevc_uint8or16.ndim` is not
    # equal to 3, the unpacking below raises a `ValueError`
    # exception.
    (height_image, width_image, nb_channels) = image_before_encoding_hevc_uint8or16.shape
    if image_before_encoding_hevc_uint8or16.dtype == numpy.uint8:
        input_bit_depth = 8
    else:
        input_bit_depth = 10
    if nb_channels == 3:
        input_chromat_format = '420'
        is_400 = False
    else:
        input_chromat_format = '400'
        is_400 = True
    
    # `encode_decode_image` is not used for collecting statistics
    # on the frequency of use of HEVC intra prediction modes.
    args_subprocess_encoding = [
        path_to_exe_encoder,
        '-c',
        path_to_cfg,
        '-i',
        path_to_before_encoding_hevc,
        '-b',
        path_to_bitstream,
        '-o',
        path_to_after_encoding_hevc,
        '-wdt',
        str(width_image),
        '-hgt',
        str(height_image),
        '--InputBitDepth={}'.format(input_bit_depth),
        '--InputChromaFormat={}'.format(input_chromat_format),
        '--FramesToBeEncoded=1',
        '--QP={}'.format(qp)
    ]
    args_subprocess_decoding = [
        path_to_exe_decoder,
        '-b',
        path_to_bitstream,
        '-o',
        path_to_after_decoding_hevc
    ]
    if dict_substitution_switch is not None:
        args_substitution_switch = [
            '--PathToAdditionalDirectory={}'.format(dict_substitution_switch['path_to_additional_directory']),
            '--PathToMeanTraining={}'.format(dict_substitution_switch['path_to_mean_training']),
            '--PathToFilePathsToGraphsOutput={}'.format(dict_substitution_switch['path_to_file_paths_to_graphs_output'])
        ]
        args_subprocess_encoding += args_substitution_switch
        args_subprocess_decoding += args_substitution_switch
    
    # A text file is opened in order to write the
    # history of the encoding via HEVC.
    with open(path_to_log_encoder, 'w') as file:
        subprocess.check_call(args_subprocess_encoding,
                              stdout=file,
                              shell=False)
    
    # Another text file is opened in order to write the
    # history of the decoding via HEVC.
    with open(path_to_log_decoder, 'w') as file:
        subprocess.check_call(args_subprocess_decoding,
                              stdout=file,
                              shell=False)
    expanded_reconstructed_image_after_decoding_hevc_uint8or16 = read_400_or_420(height_image,
                                                                                 width_image,
                                                                                 1,
                                                                                 image_before_encoding_hevc_uint8or16.dtype,
                                                                                 is_400,
                                                                                 path_to_after_decoding_hevc)
    os.remove(path_to_bitstream)
    os.remove(path_to_before_encoding_hevc)
    os.remove(path_to_after_encoding_hevc)
    os.remove(path_to_after_decoding_hevc)
    return numpy.squeeze(expanded_reconstructed_image_after_decoding_hevc_uint8or16, axis=3)

def encode_image(image_before_encoding_hevc_uint8or16, path_to_before_encoding_hevc, path_to_after_encoding_hevc,
                 path_to_cfg, path_to_bitstream, path_to_exe_encoder, qp, dict_substitution_switch, path_to_stats='',
                 path_to_thresholded_map_modes_luminance=''):
    """Encodes the image via HEVC.
    
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
    path_to_cfg : str
        Path to the configuration file. `path_to_cfg` ends
        with ".cfg".
    path_to_bitstream : str
        Path to the bitstream file. `path_to_bitstream` ends
        with ".bin".
    path_to_exe_encoder : str
        Path to the executable of the HEVC encoder.
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
    path_to_stats : str, optional
        Path to the text file containing statistics about
        intra prediction modes in HEVC. The path ends with
        ".txt". The defaut value is ''. By default, the text
        file is not created.
    path_to_thresholded_map_modes_luminance : str, optional
        Path to the saved thresholded map of intra prediction
        modes for the luminance channel. The path ends with
        ".ppm". The default value is ''. By default, the
        thresholded map is not saved.
    
    Returns
    -------
    numpy.ndarray
        Array with the same data-type and the same shape as
        `image_before_encoding_hevc_uint8or16`.
        Reconstructed image after the encoding via HEVC.
    
    """
    # The expansion of dimension below means that the image
    # is a video with a single frame.
    # `write_400_or_420` checks that `image_before_encoding_hevc_uint8or16.shape[2]`
    # is equal to either 1 or 3. `write_400_or_420` also
    # checks that `image_before_encoding_hevc_uint8or16.dtype`
    # is equal to either `numpy.uint8` or `numpy.uint16`.
    write_400_or_420(numpy.expand_dims(image_before_encoding_hevc_uint8or16, 3),
                     path_to_before_encoding_hevc)
    (height_image, width_image, nb_channels) = image_before_encoding_hevc_uint8or16.shape
    if image_before_encoding_hevc_uint8or16.dtype == numpy.uint8:
        input_bit_depth = 8
    else:
        input_bit_depth = 10
    if nb_channels == 3:
        input_chromat_format = '420'
        is_400 = False
    else:
        input_chromat_format = '400'
        is_400 = True
    args_subprocess = [
        path_to_exe_encoder,
        '-c',
        path_to_cfg,
        '-i',
        path_to_before_encoding_hevc,
        '-b',
        path_to_bitstream,
        '-o',
        path_to_after_encoding_hevc,
        '-wdt',
        str(width_image),
        '-hgt',
        str(height_image),
        '--InputBitDepth={}'.format(input_bit_depth),
        '--InputChromaFormat={}'.format(input_chromat_format),
        '--FramesToBeEncoded=1',
        '--QP={}'.format(qp),
        '--PathToStats={}'.format(path_to_stats),
        '--PathToThresholdedMapModesLuminance={}'.format(path_to_thresholded_map_modes_luminance)
    ]
    
    # If `dict_substitution_switch` is None, HEVC includes no
    # prediction neural network.
    if dict_substitution_switch is not None:
        args_subprocess += [
            '--PathToAdditionalDirectory={}'.format(dict_substitution_switch['path_to_additional_directory']),
            '--PathToMeanTraining={}'.format(dict_substitution_switch['path_to_mean_training']),
            '--PathToFilePathsToGraphsOutput={}'.format(dict_substitution_switch['path_to_file_paths_to_graphs_output'])
        ]
    
    # Setting `shell` to True makes the program
    # vulnerable to shell injection, see
    # <https://docs.python.org/2/library/subprocess.html>.
    subprocess.check_call(args_subprocess,
                          shell=False)
    expanded_reconstructed_image_after_encoding_hevc_uint8or16 = read_400_or_420(height_image,
                                                                                 width_image,
                                                                                 1,
                                                                                 image_before_encoding_hevc_uint8or16.dtype,
                                                                                 is_400,
                                                                                 path_to_after_encoding_hevc)
    os.remove(path_to_bitstream)
    os.remove(path_to_before_encoding_hevc)
    os.remove(path_to_after_encoding_hevc)
    
    # The reduction of dimension below means that the
    # video with a single frame is as an image.
    return numpy.squeeze(expanded_reconstructed_image_after_encoding_hevc_uint8or16, axis=3)

def read_400_or_420(height_video, width_video, nb_frames, data_type, is_400, path_to_video):
    """Reads either a luminance video in 4:0:0 or a YCbCr video in 4:2:0 from a binary file.
    
    Parameters
    ----------
    height_video : int
        Height of the video.
    width_video : int
        Width of the video.
    nb_frames : int
        Number of video frames to be read.
    data_type : type
        Data type of the video. `data_type` is
        equal to either `numpy.uint8` or `numpy.uint16`.
    is_400 : bool
        Is the video to be read a luminance video in 4:0:0?
        If False, the video to be read is a YCbCr video in
        4:2:0.
    path_to_video : str
        Path to the binary file from which the video is
        read. `path_to_video` ends with ".yuv".
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `data_type`.
        Video. If `is_400` is True, the array shape is equal
        to (`height_video`, `width_video`, 1, `nb_frames`).
        If `is_400` is False, the array shape is equal to
        (`height_video`, `width_video`, 3, `nb_frames`).
    
    Raises
    ------
    TypeError
        If `data_type` is equal to neither `numpy.uint8`
        nor `numpy.uint16`.
    ValueError
        If `height_video` is not divisible by 2.
    ValueError
        If `width_video` is not divisible by 2.
    
    """
    if data_type != numpy.uint8 and data_type != numpy.uint16:
        raise TypeError('`data_type` is equal to neither `numpy.uint8` nor `numpy.uint16`.')
    if height_video % 2 != 0:
        raise ValueError('`height_video` is not divisible by 2.')
    if width_video % 2 != 0:
        raise ValueError('`width_video` is not divisible by 2.')
    nb_pixels_per_frame = height_video*width_video
    nb_pixels_per_frame_divided_by_4 = nb_pixels_per_frame//4
    
    # A YCbCr video has 3 channels.
    if is_400:
        nb_channels = 1
    else:
        nb_channels = 3
    video_uint8or16 = numpy.zeros((height_video, width_video, nb_channels, nb_frames),
                                  dtype=data_type)
    
    # If `height_video` is different from the true
    # height of the video or `width_video` is different
    # from the true width of the video, there is no way
    # to systematically detect the mistake.
    with open(path_to_video, 'rb') as file:
        for i in range(nb_frames):
            y_1d_uint8or16 = numpy.fromfile(file,
                                            dtype=data_type,
                                            count=nb_pixels_per_frame)
            video_uint8or16[:, :, 0, i] = numpy.reshape(y_1d_uint8or16,
                                                        (height_video, width_video))
            if not is_400:
                cb_1d_uint8or16 = numpy.fromfile(file,
                                                 dtype=data_type,
                                                 count=nb_pixels_per_frame_divided_by_4)
                cb_2d_uint8or16 = numpy.reshape(cb_1d_uint8or16,
                                                (height_video//2, width_video//2))
                video_uint8or16[:, :, 1, i] = numpy.repeat(numpy.repeat(cb_2d_uint8or16, 2, axis=0),
                                                           2,
                                                           axis=1)
                cr_1d_uint8or16 = numpy.fromfile(file,
                                                 dtype=data_type,
                                                 count=nb_pixels_per_frame_divided_by_4)
                cr_2d_uint8or16 = numpy.reshape(cr_1d_uint8or16,
                                                (height_video//2, width_video//2))
                video_uint8or16[:, :, 2, i] = numpy.repeat(numpy.repeat(cr_2d_uint8or16, 2, axis=0),
                                                           2,
                                                           axis=1)
    return video_uint8or16

def write_400_or_420(video_uint8or16, path_to_video):
    """Writes either a luminance video in 4:0:0 or a YCbCr video in 4:2:0 to a binary file.
    
    Parameters
    ----------
    video_uint8or16 : numpy.ndarray
        4D array with data-type either `numpy.uint8` or `numpy.uint16`.
        Video. If `video_uint8or16.shape[2]` is equal to
        1, the video is considered as a luminance video.
        If `video_uint8or16.shape[2]` is equal to 3, the
        video is considered as a YCbCr video.
    path_to_video : str
        Path to binary file in which the video is saved.
        `path_to_video` ends with ".yuv".
    
    Raises
    ------
    TypeError
        If `video_uint8or16.dtype` is equal to neither `numpy.uint8`
        nor `numpy.uint16`.
    ValueError
        If `video_uint8or16.shape[0]` is not divisible by 2.
    ValueError
        If `video_uint8or16.shape[1]` is not divisible by 2.
    ValueError
        If `video_uint8or16.shape[2]` is not equal to neither 1 nor 3.
    IOError
        If a file at `path_to_video` already exists.
    
    """
    if video_uint8or16.dtype != numpy.uint8 and video_uint8or16.dtype != numpy.uint16:
        raise TypeError('`video_uint8or16.dtype` is equal to neither `numpy.uint8` nor `numpy.uint16`.')
    
    # If `video_uint8or16.ndim` is not equal to 4, the
    # unpacking below raises a `ValueError` exception.
    (height_video, width_video, nb_channels, nb_frames) = video_uint8or16.shape
    if height_video % 2 != 0:
        raise ValueError('`video_uint8or16.shape[0]` is not divisible by 2.')
    if width_video % 2 != 0:
        raise ValueError('video_uint8or16.shape[1]` is not divisible by 2.')
    if nb_channels not in (1, 3):
        raise ValueError('`video_uint8or16.shape[2]` is not equal to neither 1 nor 3.')
    
    # Another program running in parallel may have already
    # created a file at `path_to_video`. Overwritting this file
    # will cause confusion between different videos.
    if os.path.isfile(path_to_video):
        raise IOError('"{}" already exists.'.format(path_to_video))
    with open(path_to_video, 'wb') as file:
        for i in range(nb_frames):
            video_uint8or16[:, :, 0, i].flatten().tofile(file)
            if nb_channels == 3:
                video_uint8or16[::2, ::2, 1, i].flatten().tofile(file)
                video_uint8or16[::2, ::2, 2, i].flatten().tofile(file)


