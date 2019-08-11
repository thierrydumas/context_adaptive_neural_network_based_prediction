"""A library that contains functions for extracting statistics about intra prediction modes in HEVC."""

import numpy

import hevc.running

# The functions are sorted in alphabetic order.

def convert_accumulations_into_ratios(accumulations_with_index_mode_int, counts_int):
    """Divides each indicator cumulated over all text files by its count.
    
    Parameters
    ----------
    accumulations_with_index_mode_int : numpy.ndarray
        3D array whose data-type is smaller than `numpy.int`
        in type hierarchy.
        Statistics depending on the HEVC intra prediction mode.
        `accumulations_with_index_mode_int[i, j, k]` is the
        indicator for the kind of statistics of index i, the mode
        of index j and the width of target patch of index k in
        {4, 8, 16, 32, 64}.
    counts_int : numpy.ndarray
        1D array whose data-type is smaller than `numpy.int`
        in type hierarchy.
        Indicators count. `counts_int[i]` is the count of the
        indicators for the width of target patch of index i in
        {4, 8, 16, 32, 64}.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.float64`.
        Indicators cumulated over all text files divided
        by their indicator count.
    
    Raises
    ------
    TypeError
        If `accumulations_with_index_mode_int.dtype` is
        not smaller than `numpy.integer` in type hierarchy.
    TypeError
        If `counts_int` is not smaller than `numpy.integer`
        in type hierarchy.
    
    """
    if not numpy.issubdtype(accumulations_with_index_mode_int.dtype, numpy.integer):
        raise TypeError('`accumulations_with_index_mode_int.dtype` is not smaller than `numpy.integer` in type hierarchy.')
    if not numpy.issubdtype(counts_int.dtype, numpy.integer):
        raise TypeError('`counts_int` is not smaller than `numpy.integer` in type hierarchy.')
    expanded_counts_int = numpy.expand_dims(numpy.expand_dims(counts_int, 0),
                                            0)
    tiled_counts_int = numpy.tile(expanded_counts_int,
                                  (accumulations_with_index_mode_int.shape[0], accumulations_with_index_mode_int.shape[1], 1))
    tiled_counts_float64 = tiled_counts_int.astype(numpy.float64)
    return accumulations_with_index_mode_int/tiled_counts_float64

def encode_luminances_extract_statistics_from_files(luminances_uint8or16, path_to_before_encoding_hevc, path_to_after_encoding_hevc,
                                                    path_to_cfg, path_to_bitstream, path_to_exe_encoder, qp, dict_substitution_switch,
                                                    paths_to_stats, pairs_beacons, beacon_run, nb_modes, nb_widths_target):
    """Encodes each luminance image via HEVC and extracts statistics about its intra prediction modes from text files.
    
    Parameters
    ----------
    luminances_uint8or16 : numpy.ndarray
        4D array with data-type either `numpy.uint8` or `numpy.uint16`.
        Luminance images. `luminances_uint8or16[i, :, :, :]`
        is the luminance image of index i. `luminances_uint8or16.shape[3]`
        is equal to 1.
    path_to_before_encoding_hevc : str
        Path to the file storing a luminance image before the encoding
        via HEVC. `path_to_before_encoding_hevc` ends with ".yuv".
    path_to_after_encoding_hevc : str
        Path to the file storing the reconstructed luminance image
        after the encoding via HEVC. `path_to_after_encoding_hevc`
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
    paths_to_stats : list
        `paths_to_stats[i]` is the path to the text file storing
        the statistics about HEVC intra prediction modes for the
        luminance image of index i. Each string ends with ".txt".
    pairs_beacons : tuple
        Each tuple in this tuple contains two strings.
        These two strings are two beacons for finding
        a specific series of statistics that depends on
        the HEVC intra prediction mode.
    beacon_run : str
        Beacon for finding the number of times the pipeline
        {fast selection, rate-distortion} selection is run
        for each width of target patch.
    nb_modes : int
        Number of intra prediction modes to be considered.
    nb_widths_target : int
        Number of widths of target patch to be considered.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.float64`.
        Frequencies depending on the HEVC intra prediction
        mode. The element at the position [i, j, k] in this
        array is the frequency associated to `pairs_beacons[i]`,
        the mode of index j and the width of target patch of
        index k in {4, 8, 16, 32, 64}.
    
    """
    for i in range(luminances_uint8or16.shape[0]):
        hevc.running.encode_image(luminances_uint8or16[i, :, :, :],
                                  path_to_before_encoding_hevc,
                                  path_to_after_encoding_hevc,
                                  path_to_cfg,
                                  path_to_bitstream,
                                  path_to_exe_encoder,
                                  qp,
                                  dict_substitution_switch,
                                  path_to_stats=paths_to_stats[i])
    (accumulations_with_index_mode_int64, accumulations_without_index_mode_int64) = \
        extract_statistics_from_files(paths_to_stats,
                                      pairs_beacons,
                                      (beacon_run,),
                                      nb_modes,
                                      nb_widths_target)
    
    # `counts_int64[i]` is the number of times the
    # pipeline {fast selection, rate-distortion selection}
    # is run for the width of target patch of index i in
    # {4, 8, 16, 32, 64}.
    counts_int64 = numpy.squeeze(accumulations_without_index_mode_int64,
                                 axis=0)
    return convert_accumulations_into_ratios(accumulations_with_index_mode_int64,
                                             counts_int64)

def extract_statistics_from_files(paths_to_stats, pairs_beacons, beacons_before_integers, nb_modes, nb_widths_target):
    """Extracts statistics about intra prediction modes in HEVC from text files.
    
    Parameters
    ----------
    paths_to_stats : list
        `paths_to_stats[i]` is the path to the text file storing
        the statistics about HEVC intra prediction modes for the
        image of index i. Each string ends with ".txt".
    pairs_beacons : tuple
        Each tuple in this tuple contains two strings.
        These two strings are two beacons for finding
        a specific series of statistics that depends on
        the HEVC intra prediction mode.
    beacons_before_integers : tuple
        Each string in this tuple is a beacon which enables
        to find a specific series of statistics that does not
        depend on the HEVC intra prediction mode.
    nb_modes : int
        Number of intra prediction modes to be considered.
    nb_widths_target : int
        Number of widths of target patch to be considered.
    
    Returns
    -------
    tuple
        numpy.ndarray
            3D array with data-type `numpy.int64`.
            Statistics depending on the HEVC intra prediction
            mode. The element at the position [i, j, k] in this
            array is the indicator associated to `pairs_beacons[i]`,
            the mode of index j and the width of target patch of
            index k in {4, 8, 16, 32, 64}. Each indicator is cumulated
            over all text files.
        numpy.ndarray
            2D array with data-type `numpy.int64`.
            Statistics that do not depend on the HEVC intra prediction
            mode. The element at the position [i, j] in this array is
            the indicator associated to `beacons_before_integers[i]`
            and the width of target patch of index j in {4, 8, 16, 32, 64}.
            Each indicator is cumulated over all text files.
    
    """
    nb_pairs_beacons = len(pairs_beacons)
    nb_beacons_before_integers = len(beacons_before_integers)
    accumulations_with_index_mode_int64 = numpy.zeros((nb_pairs_beacons, nb_modes, nb_widths_target),
                                                      dtype=numpy.int64)
    accumulations_without_index_mode_int64 = numpy.zeros((nb_beacons_before_integers, nb_widths_target),
                                                         dtype=numpy.int64)
    containers_0_int64 = numpy.zeros((nb_pairs_beacons, nb_modes, nb_widths_target),
                                     dtype=numpy.int64)
    containers_1_int64 = numpy.zeros((nb_beacons_before_integers, nb_widths_target),
                                     dtype=numpy.int64)
    for path_to_stats in paths_to_stats:
        with open(path_to_stats, 'r') as file:
            for line_text in file:
                search_1st_match_beacons(line_text,
                                         pairs_beacons,
                                         beacons_before_integers,
                                         containers_0_int64,
                                         containers_1_int64)
        
        # The data-type of `accumulations_with_index_mode_int64` and
        # `accumulations_without_index_mode_int64` is `numpy.int64` as
        # a cumulated indicator will never reach 2**63 - 1.
        accumulations_with_index_mode_int64 += containers_0_int64
        accumulations_without_index_mode_int64 += containers_1_int64
        
        # All containers are reset to their initial
        # state before moving on to the next text file.
        containers_0_int64.fill(0)
        containers_1_int64.fill(0)
    return (accumulations_with_index_mode_int64, accumulations_without_index_mode_int64)

def fill_if_beacon_found(line_text, beacon_before_integers, container_1d_int64):
    """Fills a container with integers extracted from a line of text if a beacon is found in this line.
    
    Parameters
    ----------
    line_text : str
        Line of text.
    beacon_before_integers : str
        Beacon located just before the integers
        in the line of text. Note that the integers
        to be extracted must be located between this
        beacon and the end of the line. The integers
        must be each separated by a space.
    container_1d_int64 : numpy.ndarray
        1D array with data-type `numpy.int64`.
        Container. Note that, if the beacon is found
        in the line of text, all the container elements
        ARE OVERWRITTEN.
    
    Returns
    -------
    bool
        Is the beacon found in the line of text?
    
    Raises
    ------
    RuntimeError
        If the beacon is found in the line of text
        but the number of extracted integers is not
        equal to `container_1d_int64.size`.
    
    """
    index_before = line_text.find(beacon_before_integers)
    if index_before == -1:
        return False
    else:
        
        # `i` is incremented by 1 every time an integer
        # is put into `container_1d_int64`.
        i = 0
        list_strings = line_text[index_before + len(beacon_before_integers):].split(' ')
        for item in list_strings:
            try:
                container_1d_int64[i] = int(item)
            except ValueError:
                continue
            i += 1
        
        # The exception below prevents `container_1d_int64`
        # from being partially filled.
        if i != container_1d_int64.size:
            raise RuntimeError('The number of extracted integers is not equal to `container_1d_int64.size`.')
        return True

def fill_if_beacons_found(line_text, beacon_before_index_row, beacon_after_index_row, container_2d_int64):
    """Fills a container row with integers extracted from a line of text if two beacons are found in this line.
    
    The index of the container row to be filled is sought
    across the line.
    
    Parameters
    ----------
    line_text : str
        Line of text.
    beacon_before_index_row : str
        Beacon located just before the index
        of the container row in the line of text.
    beacon_after_index_row : str
        Beacon located between the index of the
        container row and the integers to be extracted
        in the line of text. Note that the integers to
        be extracted must be located between this beacon
        and the end of the line. The integers must be each
        separated by a space.
    container_2d_int64 : numpy.ndarray
        2D array with data-type `numpy.int64`.
        Container. Note that, if the two beacons are
        found in the line of text, an entire container
        row IS OVERWRITTEN.
    
    Returns
    -------
    bool
        Are the two beacons found in the line of text?
    
    Raises
    ------
    RuntimeError
        If the two beacons are found in the line of text
        but the number of extracted integers is not equal
        to `container_2d_int64.shape[1]`.
    
    """
    index_after = line_text.find(beacon_after_index_row)
    if index_after == -1:
        return False
    else:
        index_before = line_text.find(beacon_before_index_row)
        if index_before == -1:
            return False
        else:
            
            # If the relative positioning of the two beacons
            # is incorrect, the slicing below raises a `ValueError`
            # exception.
            index_mode = int(line_text[index_before + len(beacon_before_index_row):index_after])
            
            # `i` is incremented by 1 every time an integer
            # is put into `container_2d_int64`.
            i = 0
            list_strings = line_text[index_after + len(beacon_after_index_row):].split(' ')
            for item in list_strings:
                try:
                    container_2d_int64[index_mode, i] = int(item)
                except ValueError:
                    continue
                i += 1
            
            # The exception below prevents `container_2d_int64`
            # from being partially filled.
            if i != container_2d_int64.shape[1]:
                raise RuntimeError('The number of extracted integers is not equal to `container_2d_int64.shape[1]`.')
            return True

def search_1st_match_beacons(line_text, pairs_beacons, beacons_before_integers, containers_0_int64, containers_1_int64):
    """Searches for the 1st match between beacon(s) and a line of text.
    
    When beacon(s) and the line of text match, the
    container associated to the beacon(s) is filled
    with integers extracted from the line.
    
    Parameters
    ----------
    line_text : str
        Line of text.
    pairs_beacons : tuple
        Each tuple in this tuple contains two strings.
        These two strings are two beacons for finding
        the index of a container row to be filled
        and the integers to be extracted.
    beacons_before_integers : tuple
        Each string in this tuple is a beacon which
        enables to find the integers to be extracted.
    containers_0_int64 : numpy.ndarray
        3D array with data-type `numpy.int64`.
        1st group of containers. If the two beacons
        in `pairs_beacons[i]` are found in the line
        of text, an entire row of `containers_0_int64[i, :, :]`
        IS OVERWRITTEN.
    containers_1_int64 : numpy.ndarray
        2D array with data-type `numpy.int64`.
        2nd group of containers. If the beacon
        `beacons_before_integers[i]` is found in the
        line of text, all the elements of `containers_1_int64[i, :]`
        ARE OVERWRITTEN.
    
    """
    for i in range(len(pairs_beacons)):
        if fill_if_beacons_found(line_text, pairs_beacons[i][0], pairs_beacons[i][1], containers_0_int64[i, :, :]):
            return
    for i in range(len(beacons_before_integers)):
        if fill_if_beacon_found(line_text, beacons_before_integers[i], containers_1_int64[i, :]):
            return


